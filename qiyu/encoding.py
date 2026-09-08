"""棋盘“语言”的词表与张量编码。"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

from .rules import BLACK, EMPTY, RED


PIECES = [EMPTY, "r", "h", "e", "a", "k", "c", "p", "R", "H", "E", "A", "K", "C", "P"]
SPECIAL = ["<RED>", "<BLACK>", "<DECIDE>"]
TOKENS = SPECIAL + PIECES
TOKEN_TO_ID = {token: index for index, token in enumerate(TOKENS)}
ID_TO_TOKEN = {index: token for token, index in TOKEN_TO_ID.items()}
SEQUENCE_LENGTH = 92  # 行棋方 + 90 个棋盘格 + 决策标记


def encode_board(board: Sequence[str], side: str) -> torch.Tensor:
    if len(board) != 90:
        raise ValueError("棋盘必须有 90 个格子")
    side_token = "<RED>" if side == RED else "<BLACK>"
    ids = [TOKEN_TO_ID[side_token]]
    ids.extend(TOKEN_TO_ID[piece] for piece in board)
    ids.append(TOKEN_TO_ID["<DECIDE>"])
    return torch.tensor(ids, dtype=torch.long)


def move_target(source: int, destination: int) -> Tuple[int, int]:
    return source, destination
