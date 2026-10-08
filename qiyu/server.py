"""零额外 Web 依赖的演示服务器。"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional

from .agent import QiYuAgent
from .deepseek import (
    DeepSeekClient,
    DeepSeekConfig,
    DeepSeekError,
    compact_board_context,
    load_env_file,
    parse_data_url,
    parse_xiangqi_fen,
)
from .rules import (
    BLACK,
    RED,
    Move,
    apply_unchecked,
    initial_board,
    legal_moves,
    move_description,
    move_notation,
    opponent,
    winner,
)
from .vision import SpecializedVisionError, XiangqiVisionRecognizer


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
MASTER_CHECKPOINT = ROOT / "artifacts" / "training_master" / "best_model.pt"
V5_CHECKPOINT = ROOT / "artifacts" / "training_v5" / "best_model.pt"
V4_CHECKPOINT = ROOT / "artifacts" / "training_v4" / "best_model.pt"
V3_CHECKPOINT = ROOT / "artifacts" / "training_v3_long" / "best_model.pt"
DEFAULT_CHECKPOINT = next(
    (path for path in (V5_CHECKPOINT, V4_CHECKPOINT, V3_CHECKPOINT, MASTER_CHECKPOINT) if path.exists()),
    ROOT / "artifacts" / "training" / "best_model.pt",
)
V3_OPENING_BOOK = ROOT / "artifacts" / "master_data_v3" / "opening_book.json"
MASTER_OPENING_BOOK = ROOT / "artifacts" / "master_data" / "opening_book.json"
DEFAULT_OPENING_BOOK = V3_OPENING_BOOK if V3_OPENING_BOOK.exists() else MASTER_OPENING_BOOK
ENV_FILE = ROOT / ".env"
MAX_REQUEST_BYTES = 12 * 1024 * 1024


CHAT_SYSTEM_PROMPT = """你是「棋语」，一位友好、自然、懂中国象棋的中文 AI 伙伴。
你可以和用户自由对话，也可以讲解当前棋局。回答要自然、清楚，默认简洁。
涉及当前棋局时，以每轮附带的「可核验局面数据」为唯一事实来源。
推荐走法必须在 legal_moves 中，面向用户优先使用「炮二平五」这类中国象棋记谱。
不确定时明说不确定，不编造棋子位置、局势或历史事实。
棋力决策和合法性由本地规则/搜索模块负责，你负责理解意图与语言表达。"""


class Game:
    def __init__(self, agent: QiYuAgent):
        self.agent = agent
        self.lock = threading.Lock()
        self.reset(RED)

    def reset(self, human_side: str = RED) -> None:
        if human_side not in {RED, BLACK}:
            raise ValueError("请选择红方或黑方")
        self.board = initial_board()
        self.side = RED
        self.human_side = human_side
        self.language_lock = getattr(self, "language_lock", threading.Lock())
        self.history = []
        self.chat_history = [
            {
                "role": "assistant",
                "content": "我是棋语助教。不知道怎么走时，可以问我‘这步怎么走’或‘为什么’。",
            }
        ]
        self.revision = getattr(self, "revision", 0) + 1
        self.last_move = None
        self.last_engine = "waiting-for-player"
        self.last_attention = [0.0] * 90
        self.last_candidates = []
        self.last_search = {}
        self.last_explanation = "红方先行。点击一个红方棋子，再点击目标位置。"

    def _clear_decision_state(self, explanation: str) -> None:
        self.last_move = None
        self.last_engine = "waiting-for-player"
        self.last_attention = [0.0] * 90
        self.last_candidates = []
        self.last_search = {}
        self.last_explanation = explanation

    def _state_unlocked(self) -> Dict:
        end = winner(self.board, self.side)
        legal = legal_moves(self.board, self.side) if end is None else []
        model_metrics = getattr(self.agent, "metadata", {}).get("metrics", {})
        moves_by_source: Dict[str, list] = {}
        for move in legal:
            moves_by_source.setdefault(str(move.src), []).append(move.dst)
        return {
            "board": self.board,
            "side": self.side,
            "human_side": self.human_side,
            "winner": end,
            "legal": moves_by_source,
            "history": self.history,
            "attention": self.last_attention,
            "candidates": self.last_candidates,
            "explanation": self.last_explanation,
            "search": self.last_search,
            "revision": self.revision,
            "last_move": self.last_move,
            "model_ready": self.agent.ready,
            "model_metrics": {
                key: model_metrics[key]
                for key in ("continuation_epoch", "legal_top1_accuracy", "legal_top3_accuracy")
                if key in model_metrics
            },
            "engine": self.last_engine,
            "device": str(self.agent.device),
            "chat": self.chat_history,
        }

    def state(self) -> Dict:
        with self.lock:
            return self._state_unlocked()

    def _apply(self, move: Move, engine: str, thinking_ms: float = 0.0) -> Dict:
        moving_side = self.side
        piece = self.board[move.src]
        captured = self.board[move.dst]
        description = move_description(self.board, move)
        self.board = apply_unchecked(self.board, move)
        self.history.append({"side": moving_side, "move": move.key, "description": description})
        self.side = opponent(moving_side)
        self.revision += 1
        transition = {
            "revision": self.revision,
            "side": moving_side,
            "source": move.src,
            "destination": move.dst,
            "piece": piece,
            "captured": captured,
            "description": description,
            "thinking_ms": round(thinking_ms, 2),
            "engine": engine,
        }
        self.last_move = transition
        self.last_engine = engine
        return transition

    def _agent_step(self) -> Optional[Dict]:
        if winner(self.board, self.side) is not None:
            return None
        started = time.perf_counter()
        decision = self.agent.decide(self.board, self.side)
        thinking_ms = (time.perf_counter() - started) * 1000
        transition = self._apply(decision.move, decision.engine, thinking_ms)
        self.last_attention = decision.attention
        self.last_candidates = decision.candidates
        self.last_search = decision.search
        self.last_explanation = decision.explanation
        transition["search"] = decision.search
        return transition

    def human_move(self, source: int, destination: int) -> Dict:
        with self.lock:
            if self.side != self.human_side:
                raise ValueError("现在轮到智能体行棋")
            chosen: Optional[Move] = next(
                (
                    move
                    for move in legal_moves(self.board, self.human_side)
                    if move.src == source and move.dst == destination
                ),
                None,
            )
            if chosen is None:
                raise ValueError("这不是当前棋子的合法走法")
            transition = self._apply(chosen, "human")
            self.last_attention = [0.0] * 90
            self.last_candidates = []
            self.last_search = {}
            self.last_explanation = "已收到你的落子。黑方智能体正在观察新局面。"
            return {"state": self._state_unlocked(), "transition": transition}

    def agent_step(self) -> Dict:
        with self.lock:
            transition = self._agent_step()
            return {"state": self._state_unlocked(), "transition": transition}

    def reset_response(self, human_side: Optional[str] = None) -> Dict:
        with self.lock:
            self.reset(human_side or self.human_side)
            transition = self._agent_step() if self.human_side == BLACK else None
            return {"state": self._state_unlocked(), "transition": transition}

    def _local_chat(self, question: str) -> Dict:
        """回答与当前棋局相关的新手问题，并为提示生成可验证的合法着。"""
        with self.lock:
            question = " ".join(question.strip().split())
            if not question:
                raise ValueError("请先输入问题")
            if len(question) > 200:
                raise ValueError("问题请控制在 200 字以内")

            lowered = question.lower()
            hint_words = ("怎么走", "走哪", "提示", "建议", "推荐", "hint", "help")
            piece_rules = {
                "车": "车沿横线或竖线行走，中间不能越过其他棋子。",
                "马": "马走‘日’字；紧贴马的第一个直格有子时会蹩马腿。",
                "炮": "炮不吃子时像车一样走；吃子时必须且只能隔一枚棋子作炮架。",
                "相": "相（象）走‘田’字，不能过河，中心有子时会塞象眼。",
                "象": "象（相）走‘田’字，不能过河，中心有子时会塞象眼。",
                "仕": "仕（士）只能在九宫内斜走一格。",
                "士": "士（仕）只能在九宫内斜走一格。",
                "帅": "帅（将）只能在九宫内横或竖走一格，不能与对方将帅照面。",
                "将": "将（帅）只能在九宫内横或竖走一格，不能与对方将帅照面。",
                "兵": "兵（卒）过河前只能向前一格；过河后可向前或左右一格，永远不能后退。",
                "卒": "卒（兵）过河前只能向前一格；过河后可向前或左右一格，永远不能后退。",
            }
            asked_piece = next((piece for piece in piece_rules if piece in question), None)
            if asked_piece and any(word in question for word in ("规则", "怎么下", "怎么走")):
                reply = piece_rules[asked_piece]
            elif any(word in lowered for word in hint_words):
                if winner(self.board, self.side) is not None:
                    reply = "棋局已经结束，可以点击‘重置棋局’再开一局。"
                else:
                    decision = self.agent.decide(self.board, self.side)
                    self.last_attention = decision.attention
                    self.last_candidates = decision.candidates
                    self.last_search = decision.search
                    self.last_explanation = decision.explanation
                    notation = move_notation(self.board, decision.move)
                    who = "红方" if self.side == RED else "黑方"
                    reply = f"当前轮到{who}。建议走“{notation}”。{decision.explanation}"
            elif "为什么" in question or "原因" in question:
                reply = self.last_explanation or "请先让我推荐一步，我再解释选择依据。"
            elif "注意力" in question or "热图" in question or "颜色" in question:
                reply = (
                    "棋盘上红色越深，代表 Transformer 分配到该位置的注意力越高；"
                    "关注度最高的五个位置会直接显示百分比。它用于解释模型关注点，不等于走子胜率。"
                )
            elif "规则" in question or "棋子" in question or "怎么下" in question:
                reply = (
                    "点击你的棋子后，金色圆点就是所有合法落点。红方先行；车走直线、马走日、"
                    "炮吃子时需隔一个炮架，帅和将不能在同一直线上照面。"
                )
            elif "轮到" in question or "局势" in question or "现在" in question:
                who = "红方" if self.side == RED else "黑方"
                role = "你" if self.side == self.human_side else "智能体"
                reply = f"当前轮到{who}，也就是{role}行棋。目前已走 {len(self.history)} 步。"
            else:
                reply = (
                    "我可以根据当前棋局推荐合法着、解释上一步决策、介绍棋子规则，"
                    "或说明注意力热图。你可以直接问：‘这步怎么走？’"
                )

            self.chat_history.extend(
                [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]
            )
            self.chat_history = self.chat_history[-50:]
            return {"state": self._state_unlocked(), "transition": None, "reply": reply}

    def chat(self, question: str, client: Optional[DeepSeekClient] = None) -> Dict:
        """DeepSeek 已配置时自由对话，否则保留可离线运行的规则助教。"""
        if client is None:
            return self._local_chat(question)
        cleaned = " ".join(question.strip().split())
        if not cleaned:
            raise ValueError("请先输入问题")
        if len(cleaned) > 4000:
            raise ValueError("单条消息不能超过 4000 个字符")
        with self.language_lock:
            with self.lock:
                board = self.board[:]
                side = self.side
                legal = [
                    f"{move.key}（{move_notation(board, move)}）"
                    for move in legal_moves(board, side)
                ]
                history = self.chat_history[-16:]
                decision_hint = self.last_explanation
            context = compact_board_context(board, side, legal)
            messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}, *history]
            messages.append({
                "role": "user",
                "content": (
                    f"【可核验局面数据】{context}\n"
                    f"【本地决策摘要】{decision_hint}\n"
                    f"【用户消息】{cleaned}"
                ),
            })
            result = client.chat(messages)
            with self.lock:
                self.chat_history.extend([
                    {"role": "user", "content": cleaned},
                    {"role": "assistant", "content": result["content"]},
                ])
                self.chat_history = self.chat_history[-50:]
                state = self._state_unlocked()
            return {
                "state": state,
                "transition": None,
                "reply": result["content"],
                "model": result["model"],
                "usage": result.get("usage", {}),
            }

    def recognize_position(
        self,
        data_url: str,
        client: Optional[DeepSeekClient],
        variants: Optional[list] = None,
        vision: Optional[XiangqiVisionRecognizer] = None,
    ) -> Dict:
        parsed_variants = []
        if variants:
            if not isinstance(variants, list) or len(variants) > 4:
                raise ValueError("方向校正图数量无效")
            for index, item in enumerate(variants):
                if not isinstance(item, dict):
                    raise ValueError("方向校正图格式无效")
                label = str(item.get("label", f"方向 {index + 1}"))[:40]
                variant_bytes, variant_type = parse_data_url(str(item.get("image", "")))
                parsed_variants.append((label, variant_bytes, variant_type))
            image_bytes, mime_type = parsed_variants[0][1], parsed_variants[0][2]
        else:
            image_bytes, mime_type = parse_data_url(data_url)
        with self.language_lock:
            if vision is not None:
                try:
                    return vision.recognize(image_bytes)
                except SpecializedVisionError:
                    # 专用模型未能定位棋盘时，仍可使用 DeepSeek 生成可编辑草稿。
                    if client is None:
                        raise
            if client is None:
                raise ValueError("识图引擎尚未就绪")
            return client.recognize_position(
                image_bytes,
                mime_type,
                image_variants=parsed_variants or None,
            )

    def load_position(self, fen: str, side_override: Optional[str] = None) -> Dict:
        board, fen_side = parse_xiangqi_fen(fen)
        side = side_override if side_override in {RED, BLACK} else fen_side
        if side not in {RED, BLACK}:
            raise ValueError("请指定当前行棋方")
        with self.lock:
            self.board = board
            self.side = side
            self.human_side = side
            self.history = []
            self.revision += 1
            self.chat_history.append({
                "role": "assistant",
                "content": "已导入图片中的残局。请先对照原图核对棋子，再问我局势或下一步。",
            })
            self.chat_history = self.chat_history[-50:]
            self._clear_decision_state("已导入并通过基础规则校验的残局。请先核对棋子位置，再继续分析或行棋。")
            return {"state": self._state_unlocked(), "transition": None}


AGENT: Optional[QiYuAgent] = None
DEEPSEEK: Optional[DeepSeekClient] = None
VISION: Optional[XiangqiVisionRecognizer] = None
GAMES: Dict[str, tuple[Game, float]] = {}
GAMES_LOCK = threading.Lock()
SESSION_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
SESSION_TTL_SECONDS = 6 * 60 * 60
MAX_SESSIONS = 256


def deepseek_status() -> Dict:
    if DEEPSEEK is None:
        return {
            "configured": False,
            "provider": "DeepSeek",
            "text_model": os.environ.get("DEEPSEEK_TEXT_MODEL", "deepseek-flash"),
            "vision_model": os.environ.get("DEEPSEEK_VISION_MODEL", "deepseek-flash"),
        }
    return DEEPSEEK.status


def save_local_api_key(api_key: str) -> None:
    cleaned = api_key.strip()
    if not 20 <= len(cleaned) <= 512 or any(character.isspace() for character in cleaned):
        raise ValueError("API Key 格式无效")
    existing = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    kept = [line for line in existing if not line.strip().startswith("DEEPSEEK_API_KEY=")]
    kept.append(f"DEEPSEEK_API_KEY={cleaned}")
    temporary = ENV_FILE.with_name(".env.tmp")
    temporary.write_text("\n".join(kept) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(ENV_FILE)
    os.environ["DEEPSEEK_API_KEY"] = cleaned


def require_deepseek() -> DeepSeekClient:
    if DEEPSEEK is None:
        raise ValueError("图片识别需要 DeepSeek API Key。请先在页面的「DeepSeek 连接」中完成设置。")
    return DEEPSEEK


def game_for_session(session_id: str) -> Game:
    """为每个浏览器会话维护独立棋局，同时共享只读模型。"""
    if not SESSION_PATTERN.fullmatch(session_id):
        raise ValueError("无效的棋局会话")
    now = time.monotonic()
    with GAMES_LOCK:
        expired = [key for key, (_, accessed) in GAMES.items() if now - accessed > SESSION_TTL_SECONDS]
        for key in expired:
            GAMES.pop(key, None)
        if session_id not in GAMES:
            if len(GAMES) >= MAX_SESSIONS:
                oldest = min(GAMES, key=lambda key: GAMES[key][1])
                GAMES.pop(oldest, None)
            GAMES[session_id] = (Game(AGENT), now)
        game, _ = GAMES[session_id]
        GAMES[session_id] = (game, now)
        return game


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def _json(self, payload: Dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _game(self) -> Game:
        return game_for_session(self.headers.get("X-Qiyu-Session", ""))

    def _is_local_request(self) -> bool:
        # Render terminates TLS in a local reverse proxy, so the socket peer can
        # look like 127.0.0.1 even for a public request. Never expose the local
        # API-key writer on a hosted Render instance.
        if os.environ.get("RENDER"):
            return False
        try:
            return ipaddress.ip_address(self.client_address[0]).is_loopback
        except ValueError:
            return False

    def do_GET(self):
        if self.path == "/api/health":
            metrics = getattr(AGENT, "metadata", {}).get("metrics", {}) if AGENT else {}
            self._json(
                {
                    "ok": True,
                    "model_ready": bool(AGENT and AGENT.ready),
                    "model": (
                        "qiyu-v5" if getattr(AGENT, "metadata", {}).get("environment", {}).get("color_rotation")
                        else "qiyu-v3-retrained" if metrics.get("continuation_epoch") else "qiyu-v3"
                    ),
                    "continuation_epoch": metrics.get("continuation_epoch"),
                    "deepseek": {
                        **deepseek_status(),
                        "local_setup": self._is_local_request(),
                    },
                    "vision": VISION.status if VISION else {
                        "ready": False,
                        "engine": "deepseek-fallback" if DEEPSEEK else "unavailable",
                    },
                }
            )
            return
        if self.path == "/api/state":
            self._json({"state": self._game().state(), "transition": None})
            return
        super().do_GET()

    def do_POST(self):
        global DEEPSEEK
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_REQUEST_BYTES:
                self._json({"error": "请求体不能超过 12 MiB"}, 413)
                return
            payload = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/settings/deepseek":
                if not self._is_local_request():
                    self._json({"error": "为保护 API Key，只能在本机保存配置"}, 403)
                    return
                save_local_api_key(str(payload.get("api_key", "")))
                config = DeepSeekConfig.from_env()
                DEEPSEEK = DeepSeekClient(config) if config else None
                self._json({"ok": True, "deepseek": {**deepseek_status(), "local_setup": True}})
                return
            game = self._game()
            if self.path == "/api/reset":
                self._json(game.reset_response(payload.get("human_side")))
            elif self.path == "/api/move":
                self._json(game.human_move(int(payload["source"]), int(payload["destination"])))
            elif self.path == "/api/agent-move":
                self._json(game.agent_step())
            elif self.path == "/api/chat":
                self._json(game.chat(str(payload.get("question", "")), DEEPSEEK))
            elif self.path == "/api/vision":
                self._json(game.recognize_position(
                    str(payload.get("image", "")),
                    DEEPSEEK,
                    payload.get("images"),
                    VISION,
                ))
            elif self.path == "/api/position":
                self._json(game.load_position(str(payload.get("fen", "")), payload.get("side")))
            else:
                self._json({"error": "未知接口"}, 404)
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, 400)
        except DeepSeekError as error:
            self._json({"error": str(error)}, 502)
        except SpecializedVisionError as error:
            self._json({"error": str(error)}, 502)

    def log_message(self, format, *args):
        print("[web] " + format % args)


def main() -> None:
    global AGENT, DEEPSEEK, VISION
    parser = argparse.ArgumentParser(description="启动棋语智能体演示")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--opening-book", type=Path, default=DEFAULT_OPENING_BOOK)
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    args = parser.parse_args()
    load_env_file(ENV_FILE)
    AGENT = QiYuAgent(args.checkpoint, args.opening_book)
    deepseek_config = DeepSeekConfig.from_env()
    DEEPSEEK = DeepSeekClient(deepseek_config) if deepseek_config else None
    try:
        VISION = XiangqiVisionRecognizer()
    except (FileNotFoundError, OSError, RuntimeError):
        VISION = None
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"棋语演示已启动：http://{args.host}:{args.port}")
    print(f"模型：{'已加载' if AGENT.ready else '未找到权重，使用教师策略'}")
    print(f"DeepSeek：{deepseek_status()['text_model'] if DEEPSEEK else '尚未配置 API Key（对话回退本地助教）'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
