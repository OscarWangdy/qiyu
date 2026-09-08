"""零额外 Web 依赖的演示服务器。"""

from __future__ import annotations

import argparse
import json
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
    opponent,
    square_name,
    winner,
)


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
MASTER_CHECKPOINT = ROOT / "artifacts" / "training_master" / "best_model.pt"
DEFAULT_CHECKPOINT = MASTER_CHECKPOINT if MASTER_CHECKPOINT.exists() else ROOT / "artifacts" / "training" / "best_model.pt"
DEFAULT_OPENING_BOOK = ROOT / "artifacts" / "master_data" / "opening_book.json"


class Game:
    def __init__(self, agent: QiYuAgent):
        self.agent = agent
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.board = initial_board()
        self.side = RED
        self.history = []
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
        moves_by_source: Dict[str, list] = {}
        for move in legal:
            moves_by_source.setdefault(str(move.src), []).append(move.dst)
        return {
            "board": self.board,
            "side": self.side,
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
            "engine": self.last_engine,
            "device": str(self.agent.device),
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
            if self.side != RED:
                raise ValueError("现在不是红方回合")
            chosen: Optional[Move] = next(
                (move for move in legal_moves(self.board, RED) if move.src == source and move.dst == destination), None
            )
            if chosen is None:
                raise ValueError(f"非法走法：{square_name(source)}-{square_name(destination)}")
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

    def reset_response(self) -> Dict:
        with self.lock:
            self.reset()
            return {"state": self._state_unlocked(), "transition": None}


GAME: Optional[Game] = None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def _json(self, payload: Dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/state":
            self._json({"state": GAME.state(), "transition": None})
            return
        super().do_GET()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/reset":
                self._json(GAME.reset_response())
            elif self.path == "/api/move":
                self._json(GAME.human_move(int(payload["source"]), int(payload["destination"])))
            elif self.path == "/api/agent-move":
                self._json(GAME.agent_step())
            else:
                self._json({"error": "未知接口"}, 404)
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, 400)

    def log_message(self, format, *args):
        print("[web] " + format % args)


def main() -> None:
    global GAME
    parser = argparse.ArgumentParser(description="启动棋语智能体演示")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--opening-book", type=Path, default=DEFAULT_OPENING_BOOK)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    GAME = Game(QiYuAgent(args.checkpoint, args.opening_book))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"棋语演示已启动：http://{args.host}:{args.port}")
    print(f"模型：{'已加载' if GAME.agent.ready else '未找到权重，使用教师策略'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
