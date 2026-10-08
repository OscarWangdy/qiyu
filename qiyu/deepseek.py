"""DeepSeek 语言/视觉层，不改变棋语的规则决策层。"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import socket
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .rules import BLACK, RED, board_to_string


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TEXT_MODEL = "deepseek-flash"
DEFAULT_VISION_MODEL = "deepseek-flash"
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024
PIECES = set("rheakcpRHEAKCP.")
PIECE_LIMITS = {"K": 1, "A": 2, "E": 2, "H": 2, "R": 2, "C": 2, "P": 5}


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
            "temperature": 0.65,
            "max_tokens": 1200,
        })
        return {"content": content, "usage": usage, "model": self.config.text_model}

    def recognize_position(self, image_bytes: bytes, mime_type: str) -> Dict:
        if mime_type not in ALLOWED_IMAGE_TYPES:
            raise ValueError("仅支持 JPEG、PNG、GIF 或 WebP 图片")
        if not image_bytes:
            raise ValueError("图片内容为空")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError("图片不能超过 8 MiB")
        encoded = base64.b64encode(image_bytes).decode("ascii")
        prompt = (
            "识别这张中国象棋残局图。只输出 JSON 对象，不要 Markdown。"
            "字段必须为 fen、side_to_move、confidence、orientation、notes。"
            "fen 必须是 10 行×9 列的中国象棋 FEN，图像上方为第 0 行，下方为第 9 行；"
            "黑方用 rheakcp，红方用 RHEAKCP，空格用数字压缩。"
            "如果棋盘在图中是旋转的，先旋转到黑方在上、红方在下的标准方向。"
            "side_to_move 只能是 red 或 black；无法判断时按图上标注，仍无标注则填 red 并在 notes 说明。"
            "confidence 为 0 到 1 的数字；orientation 用简短中文说明红黑方向；"
            "notes 是中文字符串数组，列出遮挡、模糊或无法确定的棋子。"
        )
        content, usage = self._completion({
            "model": self.config.vision_model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{encoded}", "detail": "original"},
                    },
                ],
            }],
            "temperature": 0.1,
            "max_tokens": 1000,
            "response_format": {"type": "json_object"},
        })
        parsed = parse_json_object(content)
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
        notes = [str(item)[:240] for item in notes[:8]]
        return {
            "fen": board_to_fen(board, side),
            "board": board,
            "side": side,
            "confidence": round(confidence, 3),
            "orientation": str(parsed.get("orientation", "已标准化为黑上红下"))[:120],
            "notes": notes,
            "usage": usage,
            "model": self.config.vision_model,
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
    if len(board) != 90 or any(piece not in PIECES for piece in board):
        raise ValueError("残局必须是由合法棋子组成的 90 格局面")
    if board.count("K") != 1 or board.count("k") != 1:
        raise ValueError("残局必须各有一枚红帅和黑将")
    for piece, limit in PIECE_LIMITS.items():
        if board.count(piece) > limit or board.count(piece.lower()) > limit:
            raise ValueError(f"残局中{piece}类棋子数量超出规则上限")


def board_to_fen(board: Sequence[str], side: str) -> str:
    validate_position(board)
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
