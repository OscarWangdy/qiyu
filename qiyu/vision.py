"""真实场景中国象棋专用视觉引擎。

推理管线与预训练模型来自 Chinese Chess Recognition：
https://github.com/TheOne1006/chinese-chess-recognition
模型发布页：https://huggingface.co/spaces/yolo12138/Chinese_Chess_Recognition
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import threading
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime
from PIL import Image, ImageOps

from .deepseek import board_to_fen, validate_detected_position


ROOT = Path(__file__).resolve().parents[1]
POSE_MODEL = ROOT / "artifacts" / "vision" / "xiangqi_pose_v6.onnx"
LAYOUT_MODEL = ROOT / "artifacts" / "vision" / "xiangqi_layout_nano_v3.onnx"

LABELS = [
    ".", "x", "K", "A", "E", "H", "R", "C", "P",
    "k", "a", "e", "h", "r", "c", "p",
]


class SpecializedVisionError(RuntimeError):
    pass


class XiangqiVisionRecognizer:
    """先检测四个棋盘角点，透视校正后对90个交叉点一次分类。"""

    def __init__(self, pose_path: Path = POSE_MODEL, layout_path: Path = LAYOUT_MODEL):
        if not pose_path.exists() or not layout_path.exists():
            raise FileNotFoundError("中国象棋专用视觉模型不完整")
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.pose = onnxruntime.InferenceSession(str(pose_path), sess_options=options)
        self.layout = onnxruntime.InferenceSession(str(layout_path), sess_options=options)
        self.lock = threading.Lock()

    @property
    def status(self) -> Dict:
        return {
            "ready": True,
            "engine": "xiangqi-vision-onnx",
            "pipeline": "四角定位 + 透视校正 + 90点分类",
        }

    @staticmethod
    def _warp_matrix(
        center: Sequence[float],
        scale: Sequence[float],
        inverse: bool = False,
    ) -> np.ndarray:
        output_width = output_height = 256
        scale_width, scale_height = scale
        if scale_width > scale_height:
            scale_height = scale_width
        else:
            scale_width = scale_height
        center = np.asarray(center, dtype=np.float32)
        scale_array = np.asarray([scale_width, scale_height], dtype=np.float32)
        source_direction = np.asarray([-scale_width * 0.5, 0.0], dtype=np.float32)
        source = np.zeros((3, 2), dtype=np.float32)
        source[0] = center
        source[1] = center + source_direction
        delta = source[0] - source[1]
        source[2] = source[1] + np.asarray([-delta[1], delta[0]], dtype=np.float32)
        destination = np.asarray([
            [output_width * 0.5, output_height * 0.5],
            [0.0, output_height * 0.5],
            [0.0, output_height],
        ], dtype=np.float32)
        if inverse:
            return cv2.getAffineTransform(destination, source)
        return cv2.getAffineTransform(source, destination)

    def _board_keypoints(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        height, width = image_rgb.shape[:2]
        center = np.asarray([height / 2.0, width / 2.0], dtype=np.float32)
        scale = np.asarray([height * 1.25, width * 1.25], dtype=np.float32)
        matrix = self._warp_matrix(center, scale)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        affine = cv2.warpAffine(image_bgr, matrix, (256, 256), flags=cv2.INTER_LINEAR)
        normalized = (
            cv2.cvtColor(affine, cv2.COLOR_BGR2RGB)
            - np.asarray([123.675, 116.28, 103.53])
        ) / np.asarray([58.395, 57.12, 57.375])
        model_input = np.expand_dims(np.transpose(normalized.astype(np.float32), (2, 0, 1)), 0)
        input_name = self.pose.get_inputs()[0].name
        simcc_x, simcc_y = self.pose.run(None, {input_name: model_input})
        x_indexes = np.argmax(simcc_x[0], axis=1)
        y_indexes = np.argmax(simcc_y[0], axis=1)
        model_points = np.stack([x_indexes / 2.0, y_indexes / 2.0], axis=1)
        scores = np.max(simcc_x[0], axis=1) * np.max(simcc_y[0], axis=1)
        inverse = self._warp_matrix(center, scale, inverse=True)
        homogeneous = np.hstack([model_points, np.ones((len(model_points), 1))])
        return homogeneous @ inverse.T, scores

    @staticmethod
    def _rectify(image_rgb: np.ndarray, points: np.ndarray) -> np.ndarray:
        if points.shape != (4, 2):
            raise SpecializedVisionError("棋盘角点数量异常")
        destination = np.asarray([
            [50, 50], [400, 50], [50, 450], [400, 450],
        ], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(points.astype(np.float32), destination)
        return cv2.warpPerspective(image_rgb, transform, (450, 500))

    def _classify(self, rectified: np.ndarray) -> Tuple[List[str], List[float]]:
        # 模型训练时使用棋盘四周各50像素的上下文。
        cropped = rectified[25:475, 25:425]
        resized = cv2.resize(cropped, (280, 315), interpolation=cv2.INTER_LINEAR)
        normalized = (
            resized - np.asarray([123.675, 116.28, 103.53])
        ) / np.asarray([58.395, 57.12, 57.375])
        model_input = np.expand_dims(np.transpose(normalized.astype(np.float32), (2, 0, 1)), 0)
        input_name = self.layout.get_inputs()[0].name
        output, = self.layout.run(None, {input_name: model_input})
        if output.shape[1:] != (90, 16):
            raise SpecializedVisionError("专用视觉模型输出维度异常")
        indexes = np.argmax(output[0], axis=-1)
        confidences = output[0, np.arange(90), indexes]
        return [LABELS[index] for index in indexes.tolist()], confidences.tolist()

    @staticmethod
    def _decode(image_bytes: bytes) -> np.ndarray:
        try:
            image = ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))).convert("RGB")
        except (OSError, ValueError) as error:
            raise SpecializedVisionError("无法解码这张图片") from error
        return np.asarray(image)

    def recognize(self, image_bytes: bytes) -> Dict:
        image_rgb = self._decode(image_bytes)
        with self.lock:
            keypoints, keypoint_scores = self._board_keypoints(image_rgb)
            if float(np.min(keypoint_scores)) < 0.08:
                raise SpecializedVisionError("未能稳定定位棋盘四角")
            rectified = self._rectify(image_rgb, keypoints)
            labels, scores = self._classify(rectified)

        board = ["." if label in {".", "x"} else label for label in labels]
        validate_detected_position(board)
        piece_scores = [score for label, score in zip(labels, scores) if label not in {".", "x"}]
        confidence = sum(piece_scores) / len(piece_scores) if piece_scores else 0.0
        notes = ["已使用中国象棋专用模型完成透视校正和90点分类。"]
        uncertain = sum(1 for label in labels if label == "x")
        if uncertain:
            notes.append(f"{uncertain} 个交叉点被标记为遮挡/背景，已按空位处理，请核对。")
        missing = []
        if board.count("K") != 1:
            missing.append("红帅")
        if board.count("k") != 1:
            missing.append("黑将")
        if missing:
            notes.append(f"未找到{'和'.join(missing)}，请在校对棋盘中补充。")
            confidence = min(confidence, 0.65)
        return {
            "fen": board_to_fen(board, "red", require_kings=False),
            "board": board,
            "side": "red",
            "confidence": round(max(0.0, min(1.0, confidence)), 3),
            "orientation": "已由棋盘四角标准化为黑上红下",
            "notes": notes,
            "usage": {},
            "model": "xiangqi-vision-onnx",
            "reviewed": True,
        }
