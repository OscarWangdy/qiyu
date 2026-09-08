"""连接模型、象棋规则环境和自然语言解释的智能体。"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .encoding import encode_board
from .heuristic import teacher_move
from .heuristic import move_score
from .model import QiYuTransformer, joint_move_log_probs, load_checkpoint
from .rules import (
    Move,
    apply_unchecked,
    board_to_string,
    legal_moves,
    move_description,
    opponent,
    parse_move,
    square_name,
)


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class Decision:
    move: Move
    confidence: float
    candidates: List[Dict]
    attention: List[float]
    explanation: str
    engine: str
    search: Dict = field(default_factory=dict)


@dataclass
class SearchConfig:
    depth: int = 2
    max_nodes: int = 160
    time_limit_ms: int = 500
    leaf_value_weight: float = 0.9
    tactical_weight: float = 0.00015
    reply_danger_weight: float = 0.00012
    candidate_limit: int = 12
    terminal_score: float = 100.0


class QiYuAgent:
    def __init__(
        self,
        checkpoint: Optional[Path] = None,
        opening_book: Optional[Path] = None,
        device: Optional[torch.device] = None,
        search_config: Optional[SearchConfig] = None,
    ):
        self.device = device or best_device()
        self.model: Optional[QiYuTransformer] = None
        self.metadata: Dict = {}
        self.opening_book: Dict[str, List[Dict]] = {}
        self.search_config = search_config or SearchConfig()
        if checkpoint and checkpoint.exists():
            self.model, self.metadata = load_checkpoint(str(checkpoint), self.device)
        if opening_book and opening_book.exists():
            self.opening_book = json.loads(opening_book.read_text(encoding="utf-8"))

    @property
    def ready(self) -> bool:
        return self.model is not None

    @torch.no_grad()
    def decide(self, board: Sequence[str], side: str) -> Decision:
        legal = legal_moves(board, side)
        if not legal:
            raise ValueError("当前局面没有合法走法")
        if self.model is None:
            move = teacher_move(board, side)
            return Decision(
                move=move,
                confidence=1.0,
                candidates=[{"move": move.key, "score": 1.0}],
                attention=[0.0] * 90,
                explanation="模型权重尚未加载，当前由可复现的教师策略完成决策。" + move_description(board, move) + "。",
                engine="teacher-fallback",
                search={"depth": 0, "nodes": 0, "elapsed_ms": 0.0, "opening_book": False},
            )

        inputs = encode_board(board, side).unsqueeze(0).to(self.device)
        source_logits, destination_logits, _, attention = self.model(inputs, capture_attention=True)
        joint_log_probs = joint_move_log_probs(source_logits, destination_logits)
        focus = attention[0].detach().cpu().tolist() if attention is not None else [0.0] * 90

        # 常见局面优先采用职业棋谱中出现次数最多的走法。
        book_key = board_to_string(board) + "|" + side
        book_entries = self.opening_book.get(book_key, [])
        book_candidates: List[Tuple[int, Move]] = []
        legal_pairs = {(move.src, move.dst): move for move in legal}
        for entry in book_entries:
            try:
                source, destination = parse_move(entry["move"])
            except ValueError:
                continue
            move = legal_pairs.get((source, destination))
            if move is not None:
                book_candidates.append((int(entry["count"]), move))
        if book_candidates:
            visits = sum(count for count, _ in book_candidates)
            move = book_candidates[0][1]
            candidates = [
                {"move": candidate.key, "score": round(count / visits, 4), "master_games": count}
                for count, candidate in book_candidates[:5]
            ]
            return Decision(
                move=move,
                confidence=book_candidates[0][0] / visits,
                candidates=candidates,
                attention=focus,
                explanation=(
                    f"当前局面命中象甲职业棋谱库，共参考 {visits} 次大师实战选择。"
                    f"最高频走法是{move_description(board, move)}；热区仍显示 Transformer 对当前局面的关注位置。"
                ),
                engine="master-opening-book",
                search={"depth": 0, "nodes": 0, "elapsed_ms": 0.0, "opening_book": True},
            )

        started = time.perf_counter()
        ranked = self._rank_moves(board, side, legal, joint_log_probs[0])
        nodes = len(legal)
        scored_moves: List[Tuple[float, Move, float]] = []
        alpha = -10_000.0
        for base_score, move, value in ranked[: self.search_config.candidate_limit]:
            if (time.perf_counter() - started) * 1000 >= self.search_config.time_limit_ms:
                scored_moves.append((base_score, move, value))
                continue
            next_board = apply_unchecked(board, move)
            score, child_nodes = self._negamax(
                next_board,
                opponent(side),
                self.search_config.depth - 1,
                -10_000.0,
                -alpha,
                started,
            )
            nodes += child_nodes
            search_score = -score + base_score
            alpha = max(alpha, search_score)
            scored_moves.append((search_score, move, value))
        for base_score, move, value in ranked[self.search_config.candidate_limit :]:
            scored_moves.append((base_score, move, value))
        scored_moves.sort(key=lambda item: item[0], reverse=True)
        score_tensor = torch.tensor([item[0] for item in scored_moves], device=self.device)
        probabilities_all = torch.softmax(score_tensor, dim=0).detach().cpu().tolist()
        elapsed_ms = (time.perf_counter() - started) * 1000
        move = scored_moves[0][1]
        top_candidates = [
            {
                "move": item[1].key,
                "score": round(probabilities_all[index], 4),
                "value": round(float(item[2]), 3),
                "search_score": round(float(item[0]), 3),
            }
            for index, item in enumerate(scored_moves[:5])
        ]
        top_focus = sorted(range(90), key=lambda index: focus[index], reverse=True)[:3]
        focus_text = "、".join(square_name(index) for index in top_focus)
        explanation = (
            f"模型在 {len(legal)} 个合法动作中进行策略排序，并用 {self.search_config.depth} 层限时搜索选择了{move_description(board, move)}。"
            f"决策时最关注的棋盘位置是 {focus_text}；界面上的红色热区显示完整注意力分布。"
        )
        return Decision(
            move=move,
            confidence=probabilities_all[0],
            candidates=top_candidates,
            attention=focus,
            explanation=explanation,
            engine="qiyu-transformer-search",
            search={
                "depth": self.search_config.depth,
                "nodes": nodes,
                "elapsed_ms": round(elapsed_ms, 2),
                "opening_book": False,
                "time_limit_ms": self.search_config.time_limit_ms,
            },
        )

    @torch.no_grad()
    def _rank_moves(
        self,
        board: Sequence[str],
        side: str,
        legal: Sequence[Move],
        joint_log_probs: torch.Tensor,
    ) -> List[Tuple[float, Move, float]]:
        policy_scores = torch.stack([joint_log_probs[move.src, move.dst] for move in legal])
        policy_log_probs = torch.log_softmax(policy_scores, dim=0)
        next_boards = [apply_unchecked(board, move) for move in legal]
        next_inputs = torch.stack([encode_board(next_board, opponent(side)) for next_board in next_boards]).to(self.device)
        _, _, opponent_values, _ = self.model(next_inputs)
        current_values = -opponent_values
        tactical = torch.tensor(
            [move_score(board, side, move) * self.search_config.tactical_weight for move in legal],
            device=self.device,
        )
        combined = policy_log_probs + self.search_config.leaf_value_weight * current_values + tactical
        probe_order = torch.argsort(combined, descending=True).tolist()[: self.search_config.candidate_limit]
        for index in probe_order:
            replies = legal_moves(next_boards[index], opponent(side))
            if not replies:
                combined[index] += self.search_config.terminal_score
                continue
            reply_danger = max(move_score(next_boards[index], opponent(side), reply) for reply in replies)
            combined[index] -= reply_danger * self.search_config.reply_danger_weight
        ranked = [
            (float(combined[index].item()), legal[index], float(current_values[index].item()))
            for index in torch.argsort(combined, descending=True).tolist()
        ]
        return ranked

    @torch.no_grad()
    def _position_value(self, board: Sequence[str], side: str) -> float:
        inputs = encode_board(board, side).unsqueeze(0).to(self.device)
        _, _, value, _ = self.model(inputs)
        return float(value[0].item())

    def _negamax(
        self,
        board: Sequence[str],
        side: str,
        depth: int,
        alpha: float,
        beta: float,
        started: float,
    ) -> Tuple[float, int]:
        if (time.perf_counter() - started) * 1000 >= self.search_config.time_limit_ms:
            return self._position_value(board, side), 1
        legal = legal_moves(board, side)
        if not legal:
            return -self.search_config.terminal_score, 1
        if depth <= 0:
            return self._position_value(board, side), 1
        nodes = 1
        best = -10_000.0
        ranked = sorted(legal, key=lambda move: move_score(board, side, move), reverse=True)
        for move in ranked[: self.search_config.candidate_limit]:
            child, child_nodes = self._negamax(
                apply_unchecked(board, move),
                opponent(side),
                depth - 1,
                -beta,
                -alpha,
                started,
            )
            nodes += child_nodes
            score = -child + move_score(board, side, move) * self.search_config.tactical_weight
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta or nodes >= self.search_config.max_nodes:
                break
        return best, nodes
