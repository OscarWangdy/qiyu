"""用于构造可复现实验数据的浅层教师策略。"""

from __future__ import annotations

import hashlib
import random
from typing import Sequence

from .rules import BLACK, EMPTY, RED, Move, apply_unchecked, board_to_string, is_in_check, legal_moves, rc


VALUES = {"K": 10000, "R": 900, "C": 450, "H": 400, "E": 200, "A": 200, "P": 120}


def move_score(board: Sequence[str], side: str, move: Move) -> float:
    piece = board[move.src]
    src_row, src_col = rc(move.src)
    dst_row, dst_col = rc(move.dst)
    score = 0.0
    if move.captured != EMPTY:
        score += VALUES[move.captured.upper()] * 10
        score -= VALUES[piece.upper()] * 0.08
    next_board = apply_unchecked(board, move)
    if is_in_check(next_board, BLACK if side == RED else RED):
        score += 260
    # 轻微鼓励向前、靠近中心和出动未发展棋子。
    forward = (src_row - dst_row) if side == RED else (dst_row - src_row)
    score += forward * 8
    score += (4 - abs(dst_col - 4)) * 2
    if piece.upper() in {"H", "C"} and src_row in {0, 2, 7, 9}:
        score += 14
    return score


def teacher_move(board: Sequence[str], side: str) -> Move:
    candidates = legal_moves(board, side)
    if not candidates:
        raise ValueError("当前局面没有合法走法")
    digest = hashlib.sha256((board_to_string(board) + side).encode("ascii")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    scored = [(move_score(board, side, move), rng.random(), move) for move in candidates]
    return max(scored, key=lambda item: (item[0], item[1]))[2]
