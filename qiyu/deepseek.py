"""DeepSeek 语言/视觉层，不改变棋语的规则决策层。"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import re
import socket
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .rules import BLACK, RED, board_to_string


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TEXT_MODEL = "deepseek-flash"
DEFAULT_VISION_MODEL = "deepseek-flash"
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024
PIECES = set("rheakcpRHEAKCP.")
PIECE_LIMITS = {"K": 1, "A": 2, "E": 2, "H": 2, "R": 2, "C": 2, "P": 5}
PIECE_TYPES = {
    "rook": "r", "chariot": "r", "车": "r", "車": "r", "俥": "r",
    "horse": "h", "knight": "h", "马": "h", "馬": "h", "傌": "h",
    "elephant": "e", "bishop": "e", "象": "e", "相": "e",
    "advisor": "a", "guard": "a", "士": "a", "仕": "a",
    "king": "k", "general": "k", "将": "k", "將": "k", "帅": "k", "帥": "k",
    "cannon": "c", "炮": "c", "砲": "c",
    "pawn": "p", "soldier": "p", "卒": "p", "兵": "p",
}


class DeepSeekError(RuntimeError):
    """对外只暴露可读错误，永不回传 API Key。"""


def load_env_file(path: Path) -> None:
    """加载本地 .env，但不覆盖已由部署环境注入的变量。"""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key.startswith("DEEPSEEK_") and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    text_model: str = DEFAULT_TEXT_MODEL
    vision_model: str = DEFAULT_VISION_MODEL
    timeout_seconds: int = 45

    @classmethod
    def from_env(cls) -> Optional["DeepSeekConfig"]:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            return None
        return cls(
            api_key=api_key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            text_model=os.environ.get("DEEPSEEK_TEXT_MODEL", DEFAULT_TEXT_MODEL),
            vision_model=os.environ.get("DEEPSEEK_VISION_MODEL", DEFAULT_VISION_MODEL),
            timeout_seconds=int(os.environ.get("DEEPSEEK_TIMEOUT", "45")),
        )


Transport = Callable[[str, Dict, Dict[str, str], int], Dict]


class DeepSeekClient:
    def __init__(self, config: DeepSeekConfig, transport: Optional[Transport] = None):
        self.config = config
        self._transport = transport or self._http_transport

    @property
    def status(self) -> Dict:
        return {
            "configured": True,
            "provider": "DeepSeek",
            "text_model": self.config.text_model,
            "vision_model": self.config.vision_model,
        }

    @staticmethod
    def _http_transport(url: str, payload: Dict, headers: Dict[str, str], timeout: int) -> Dict:
        request = Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = ""
            try:
                body = json.loads(error.read().decode("utf-8"))
                detail = body.get("error", {}).get("message", "")
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                pass
            if error.code == 401:
                raise DeepSeekError("DeepSeek API Key 无效或已失效。") from error
            if error.code == 402:
                raise DeepSeekError("DeepSeek 账户余额不足，请在开放平台查看用量。") from error
            if error.code == 429:
                raise DeepSeekError("DeepSeek 请求过于频繁，请稍后再试。") from error
            suffix = f"：{detail[:180]}" if detail else ""
            raise DeepSeekError(f"DeepSeek 服务返回 {error.code}{suffix}") from error
        except (URLError, socket.timeout, TimeoutError) as error:
            raise DeepSeekError("无法连接 DeepSeek 服务，请检查网络后重试。") from error
        except json.JSONDecodeError as error:
            raise DeepSeekError("DeepSeek 返回了无法解析的响应。") from error

    def _completion(self, payload: Dict) -> Tuple[str, Dict]:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        result = self._transport(
            f"{self.config.base_url}/chat/completions",
            payload,
            headers,
            self.config.timeout_seconds,
        )
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise DeepSeekError("DeepSeek 响应缺少正文。") from error
        if not isinstance(content, str) or not content.strip():
            raise DeepSeekError("DeepSeek 未返回有效内容。")
        return content.strip(), result.get("usage", {})

    def chat(self, messages: Sequence[Dict[str, str]]) -> Dict:
        content, usage = self._completion({
            "model": self.config.text_model,
            "messages": list(messages),
            "thinking": {"type": "disabled"},
            "temperature": 0.65,
            "max_tokens": 1200,
        })
        return {"content": content, "usage": usage, "model": self.config.text_model}

    def recognize_position(
        self,
        image_bytes: bytes,
        mime_type: str,
        image_variants: Optional[Sequence[Tuple[str, bytes, str]]] = None,
    ) -> Dict:
        media = list(image_variants or [("原图", image_bytes, mime_type)])
        if not media or len(media) > 4:
            raise ValueError("识图请求必须包含 1 到 4 张方向校正图")
        for _, variant_bytes, variant_type in media:
            if variant_type not in ALLOWED_IMAGE_TYPES:
                raise ValueError("仅支持 JPEG、PNG、GIF 或 WebP 图片")
            if not variant_bytes:
                raise ValueError("图片内容为空")
            if len(variant_bytes) > MAX_IMAGE_BYTES:
                raise ValueError("单张图片不能超过 8 MiB")

        # 对同一张图同时提供多个旋转版本会使视觉模型重复计数。
        # 只使用用户原图，由模型在单一坐标系中判定方向。
        media = media[:1]
        prompt = (
            "你是中国象棋棋盘视觉测量器，不是残局推理器。"
            "先定位主棋盘四角和 9 列×10 行的90个网格交叉点。"
            "对每一枚圆形棋子，先以棋子圆心找到最近的交叉点，再读颜色和汉字；"
            "最后才把坐标标准化为黑方在上、红方在下的 row=0..9、col=0..8。"
            "只认主棋盘最外层网格线以内、圆心贴近交叉点的棋子；"
            "必须忽略红框外、棋盘旁和盒子里堆放的已吃棋子。"
            "完全禁止根据常见阵型、棋理或‘双方应各有一将’补棋、移棋、改色。图中没有帅或将就如实缺失。"
            "逐枚检查后写入 pieces；同一格最多一子。"
            "type 只能是 rook、horse、elephant、advisor、king、cannon、pawn，color 只能是 red 或 black。"
            "同时输出 grid_corners、fen、pieces、side_to_move、confidence、orientation、notes 七个字段，只输出 JSON 对象。"
            "grid_corners 是原图像素坐标，按标准化后的 top_left、top_right、bottom_right、bottom_left 输出，"
            "例如 {\"top_left\":[70,50],\"top_right\":[410,50],\"bottom_right\":[410,470],\"bottom_left\":[70,470]}。"
            "pieces 中每枚子都要带所选图片上的像素圆心 x、y，用它自查是否映射到最近交叉点。"
            "pieces 示例：[{\"x\":210,\"y\":69,\"row\":0,\"col\":4,\"color\":\"black\",\"type\":\"king\",\"glyph\":\"将\",\"confidence\":0.95}]。"
            "fen 必须是同一 pieces 列表对应的 10 行×9 列中国象棋 FEN；"
            "黑方用 rheakcp，红方用 RHEAKCP，空格用数字压缩。"
            "side_to_move 只能是 red 或 black；无法判断时按图上标注，仍无标注则填 red 并在 notes 说明。"
            "confidence 为 0 到 1 的数字；orientation 用简短中文说明红黑方向；"
            "notes 是中文字符串数组，列出遮挡、模糊、方向选择或无法确定的棋子。"
            "输出前逐项核对：圆心是否在主棋盘内、最近网格行列、红黑颜色、棋子汉字以及 pieces 与 fen 完全一致。"
        )

        def image_content(
            instruction: str,
            attachments: Optional[Sequence[Tuple[str, bytes, str]]] = None,
        ) -> List[Dict]:
            parts: List[Dict] = [{"type": "text", "text": instruction}]
            for label, variant_bytes, variant_type in attachments or media:
                encoded = base64.b64encode(variant_bytes).decode("ascii")
                parts.extend([
                    {"type": "text", "text": f"【{label}】"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{variant_type};base64,{encoded}",
                            "detail": "original",
                        },
                    },
                ])
            return parts

        content, usage = self._completion({
            "model": self.config.vision_model,
            # Flash 默认开启思考模式；残局图可能把较小的输出预算全部
            # 消耗在 reasoning_content，最终 content 为空。结构化提取直接
            # 使用非思考模式，既稳定又能减少延迟与费用。
            "thinking": {"type": "disabled"},
            "messages": [{
                "role": "user",
                "content": image_content(prompt),
            }],
            "temperature": 0.1,
            "max_tokens": 1800,
            "response_format": {"type": "json_object"},
        })

        first_content = content
        first_parsed = parse_json_object(first_content)
        crop_media = build_piece_crops(image_bytes, first_parsed)
        review_prompt = (
            "你是第二位独立复核员。重新逐格观察同一张原图，不要盲从初稿。"
            "重点纠正：旋转方向、相邻行列错位、书法体误判，以及把棋盘外散子纳入局面。"
            "用每枚子的像素圆心复查最近交叉点。只记录主棋盘网格交叉点上确实可见的棋子，"
            "不按规则臆造或移动棋子；不得为了凑齐将帅而新增棋子。"
            "原图后如果附有‘候选棋子特写’，它们是按初稿圆心自动裁切和放大的，"
            "只用于核对该坐标的颜色与汉字，不得将同一特写重复计数。"
            "按与初稿完全相同的七字段 JSON 结构返回修正版，并保证 pieces 与 fen 一致。\n"
            f"【待复核初稿】{first_content[:6000]}"
        )
        review_content, review_usage = self._completion({
            "model": self.config.vision_model,
            "thinking": {"type": "disabled"},
            "messages": [{
                "role": "user",
                "content": image_content(review_prompt, [media[0], *crop_media]),
            }],
            "temperature": 0.1,
            "max_tokens": 1800,
            "response_format": {"type": "json_object"},
        })
        review_parsed = parse_json_object(review_content)
        first_count = len(first_parsed.get("pieces", [])) if isinstance(first_parsed.get("pieces"), list) else 0
        review_count = len(review_parsed.get("pieces", [])) if isinstance(review_parsed.get("pieces"), list) else 0
        # 复核的目标是纠错，不是把大部分已检出棋子删掉。
        content = first_content if first_count >= 3 and review_count < first_count * 0.7 else review_content
        usage = {"first_pass": usage, "review_pass": review_usage}

        parsed = parse_json_object(content)
        board: Optional[List[str]] = None
        parser_notes: List[str] = []
        pieces = parsed.get("pieces")
        if isinstance(pieces, list):
            candidate = ["."] * 90
            candidate_confidence = [-1.0] * 90
            for item in pieces:
                if not isinstance(item, dict):
                    parser_notes.append("已忽略一条格式无效的棋子候选。")
                    continue
                try:
                    row, col = int(item["row"]), int(item["col"])
                except (KeyError, TypeError, ValueError):
                    parser_notes.append("已忽略一枚缺少有效行列的棋子候选。")
                    continue
                if not (0 <= row < 10 and 0 <= col < 9):
                    parser_notes.append(f"已忽略坐标越界的棋子（{row}, {col}）。")
                    continue
                piece_type = PIECE_TYPES.get(str(item.get("type", "")).strip().lower())
                if not piece_type:
                    piece_type = PIECE_TYPES.get(str(item.get("glyph", "")).strip())
                color = str(item.get("color", "")).strip().lower()
                if not piece_type or color not in {RED, BLACK}:
                    parser_notes.append(f"已忽略 {row + 1} 行 {col + 1} 列的未知棋子候选。")
                    continue
                index = row * 9 + col
                try:
                    item_confidence = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
                except (TypeError, ValueError):
                    item_confidence = 0.5
                symbol = piece_type.upper() if color == RED else piece_type
                if candidate[index] != ".":
                    parser_notes.append(f"{row + 1} 行 {col + 1} 列出现多枚候选，已暂留高置信度的一枚，请手动核对。")
                    if item_confidence <= candidate_confidence[index]:
                        continue
                candidate[index] = symbol
                candidate_confidence[index] = item_confidence
            validate_detected_position(candidate)
            board = candidate

        fen_side: Optional[str] = None
        if board is None:
            board, fen_side = parse_xiangqi_fen(str(parsed.get("fen", "")))
        requested_side = str(parsed.get("side_to_move", "")).lower()
        side = requested_side if requested_side in {RED, BLACK} else fen_side
        if side not in {RED, BLACK}:
            side = RED
        confidence = parsed.get("confidence", 0.0)
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.0
        notes = parsed.get("notes", [])
        if not isinstance(notes, list):
            notes = [str(notes)]
        notes = [str(item)[:240] for item in notes[:8]] + parser_notes[:8]
        missing_kings = []
        if board.count("K") != 1:
            missing_kings.append("红帅")
        if board.count("k") != 1:
            missing_kings.append("黑将")
        if missing_kings:
            missing_text = "和".join(missing_kings)
            notes.append(f"识别草稿未找到{missing_text}；请在校对棋盘中补充后再应用。")
            confidence = min(confidence, 0.65 if len(missing_kings) == 1 else 0.45)
        return {
            "fen": board_to_fen(board, side, require_kings=False),
            "board": board,
            "side": side,
            "confidence": round(confidence, 3),
            "orientation": str(parsed.get("orientation", "已标准化为黑上红下"))[:120],
            "notes": notes,
            "usage": usage,
            "model": self.config.vision_model,
            "reviewed": True,
        }


def parse_json_object(text: str) -> Dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as error:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise DeepSeekError("图片识别结果不是有效 JSON。") from error
        try:
            value = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as nested_error:
            raise DeepSeekError("图片识别结果不是有效 JSON。") from nested_error
    if not isinstance(value, dict):
        raise DeepSeekError("图片识别结果必须是 JSON 对象。")
    return value


def build_piece_crops(image_bytes: bytes, parsed: Dict) -> List[Tuple[str, bytes, str]]:
    """根据第一遍的圆心坐标裁出特写，让复核模型看清低分辨率汉字。"""
    pieces = parsed.get("pieces")
    if not isinstance(pieces, list):
        return []
    try:
        image = ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))).convert("RGB")
    except (OSError, ValueError):
        return []

    corners = parsed.get("grid_corners", {})
    points = []
    if isinstance(corners, dict):
        for name in ("top_left", "top_right", "bottom_right", "bottom_left"):
            point = corners.get(name)
            if isinstance(point, list) and len(point) == 2:
                try:
                    points.append((float(point[0]), float(point[1])))
                except (TypeError, ValueError):
                    points = []
                    break
    if len(points) == 4:
        width = (abs(points[1][0] - points[0][0]) + abs(points[2][0] - points[3][0])) / 2
        height = (abs(points[3][1] - points[0][1]) + abs(points[2][1] - points[1][1])) / 2
        radius = int(max(20, min(70, max(width / 8, height / 9) * 0.72)))
    else:
        radius = int(max(20, min(60, min(image.size) / 13)))

    crops: List[Tuple[str, bytes, str]] = []
    seen_centers = set()
    for item in pieces:
        if not isinstance(item, dict):
            continue
        try:
            x, y = round(float(item["x"])), round(float(item["y"]))
            row, col = int(item["row"]), int(item["col"])
        except (KeyError, TypeError, ValueError):
            continue
        center_key = (x // 4, y // 4)
        if center_key in seen_centers or not (0 <= x < image.width and 0 <= y < image.height):
            continue
        seen_centers.add(center_key)
        box = (
            max(0, x - radius), max(0, y - radius),
            min(image.width, x + radius), min(image.height, y + radius),
        )
        crop = image.crop(box).resize((256, 256), Image.Resampling.LANCZOS)
        crop = ImageEnhance.Contrast(crop).enhance(1.12)
        crop = crop.filter(ImageFilter.UnsharpMask(radius=1.4, percent=130, threshold=3))
        output = BytesIO()
        crop.save(output, format="JPEG", quality=92)
        crops.append((f"候选棋子特写：第 {row + 1} 行第 {col + 1} 列", output.getvalue(), "image/jpeg"))
        if len(crops) >= 12:
            break
    return crops


def parse_data_url(data_url: str) -> Tuple[bytes, str]:
    match = re.fullmatch(r"data:([^;,]+);base64,([A-Za-z0-9+/=\s]+)", data_url or "")
    if not match:
        raise ValueError("图片数据格式无效")
    mime_type = match.group(1).lower()
    if mime_type not in ALLOWED_IMAGE_TYPES:
        raise ValueError("仅支持 JPEG、PNG、GIF 或 WebP 图片")
    try:
        image_bytes = base64.b64decode(match.group(2), validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("图片 Base64 数据无效") from error
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("图片不能超过 8 MiB")
    return image_bytes, mime_type


def parse_xiangqi_fen(fen: str) -> Tuple[List[str], Optional[str]]:
    fields = fen.strip().split()
    if not fields:
        raise ValueError("残局 FEN 为空")
    rows = fields[0].split("/")
    if len(rows) != 10:
        raise ValueError("残局 FEN 必须包含 10 行")
    board: List[str] = []
    aliases = {"n": "h", "N": "H", "b": "e", "B": "E"}
    for row in rows:
        cells: List[str] = []
        for symbol in row:
            if symbol.isdigit() and symbol != "0":
                cells.extend(["."] * int(symbol))
            else:
                symbol = aliases.get(symbol, symbol)
                if symbol not in PIECES or symbol == ".":
                    raise ValueError(f"残局 FEN 包含未知棋子：{symbol}")
                cells.append(symbol)
        if len(cells) != 9:
            raise ValueError("残局 FEN 每行必须恰好包含 9 格")
        board.extend(cells)
    validate_position(board)
    side: Optional[str] = None
    if len(fields) > 1:
        marker = fields[1].lower()
        side = RED if marker in {"w", "r", "red"} else BLACK if marker in {"b", "black"} else None
    return board, side


def validate_position(board: Sequence[str]) -> None:
    validate_detected_position(board)
    if board.count("K") != 1 or board.count("k") != 1:
        raise ValueError("残局必须各有一枚红帅和黑将")


def validate_detected_position(board: Sequence[str]) -> None:
    """校验视觉草稿，不强迫模型臆造未拍到的将帅。"""
    if len(board) != 90 or any(piece not in PIECES for piece in board):
        raise ValueError("残局必须是由合法棋子组成的 90 格局面")
    for piece, limit in PIECE_LIMITS.items():
        if board.count(piece) > limit or board.count(piece.lower()) > limit:
            raise ValueError(f"残局中{piece}类棋子数量超出规则上限")


def board_to_fen(board: Sequence[str], side: str, require_kings: bool = True) -> str:
    (validate_position if require_kings else validate_detected_position)(board)
    rows: List[str] = []
    for row_index in range(10):
        text, empty = "", 0
        for symbol in board[row_index * 9:(row_index + 1) * 9]:
            if symbol == ".":
                empty += 1
            else:
                if empty:
                    text += str(empty)
                    empty = 0
                text += symbol
        if empty:
            text += str(empty)
        rows.append(text)
    return "/".join(rows) + (" w" if side == RED else " b")


def compact_board_context(board: Sequence[str], side: str, legal_keys: Sequence[str]) -> str:
    return json.dumps({
        "coordinate_system": "左上 a0，右下 i9；黑方在上，红方在下",
        "board_90": board_to_string(board),
        "side_to_move": side,
        "legal_moves": list(legal_keys),
    }, ensure_ascii=False)
