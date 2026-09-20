"""零额外 Web 依赖的演示服务器。"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional

from .agent import QiYuAgent
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


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
MASTER_CHECKPOINT = ROOT / "artifacts" / "training_master" / "best_model.pt"
V4_CHECKPOINT = ROOT / "artifacts" / "training_v4" / "best_model.pt"
V3_CHECKPOINT = ROOT / "artifacts" / "training_v3_long" / "best_model.pt"
DEFAULT_CHECKPOINT = next(
    (path for path in (V4_CHECKPOINT, V3_CHECKPOINT, MASTER_CHECKPOINT) if path.exists()),
    ROOT / "artifacts" / "training" / "best_model.pt",
)
V3_OPENING_BOOK = ROOT / "artifacts" / "master_data_v3" / "opening_book.json"
MASTER_OPENING_BOOK = ROOT / "artifacts" / "master_data" / "opening_book.json"
DEFAULT_OPENING_BOOK = V3_OPENING_BOOK if V3_OPENING_BOOK.exists() else MASTER_OPENING_BOOK


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

    def chat(self, question: str) -> Dict:
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


AGENT: Optional[QiYuAgent] = None
GAMES: Dict[str, tuple[Game, float]] = {}
GAMES_LOCK = threading.Lock()
SESSION_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
SESSION_TTL_SECONDS = 6 * 60 * 60
MAX_SESSIONS = 256


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

    def do_GET(self):
        if self.path == "/api/health":
            metrics = getattr(AGENT, "metadata", {}).get("metrics", {}) if AGENT else {}
            self._json(
                {
                    "ok": True,
                    "model_ready": bool(AGENT and AGENT.ready),
                    "model": "qiyu-v3-retrained" if metrics.get("continuation_epoch") else "qiyu-v3",
                    "continuation_epoch": metrics.get("continuation_epoch"),
                }
            )
            return
        if self.path == "/api/state":
            self._json({"state": self._game().state(), "transition": None})
            return
        super().do_GET()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            game = self._game()
            if self.path == "/api/reset":
                self._json(game.reset_response(payload.get("human_side")))
            elif self.path == "/api/move":
                self._json(game.human_move(int(payload["source"]), int(payload["destination"])))
            elif self.path == "/api/agent-move":
                self._json(game.agent_step())
            elif self.path == "/api/chat":
                self._json(game.chat(str(payload.get("question", ""))))
            else:
                self._json({"error": "未知接口"}, 404)
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, 400)

    def log_message(self, format, *args):
        print("[web] " + format % args)


def main() -> None:
    global AGENT
    parser = argparse.ArgumentParser(description="启动棋语智能体演示")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--opening-book", type=Path, default=DEFAULT_OPENING_BOOK)
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    args = parser.parse_args()
    AGENT = QiYuAgent(args.checkpoint, args.opening_book)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"棋语演示已启动：http://{args.host}:{args.port}")
    print(f"模型：{'已加载' if AGENT.ready else '未找到权重，使用教师策略'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
