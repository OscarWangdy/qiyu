"""轻量、无第三方依赖的中国象棋规则环境。

内部使用 0–89 的一维下标；面向使用者的文案一律使用中国象棋记谱。
小写棋子为黑方，大写棋子为红方。实现覆盖项目演示所需的走法、
蹩马腿、塞象眼、炮架、九宫、过河兵、将帅照面、自将检测与终局判断。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


ROWS, COLS = 10, 9
RED, BLACK = "red", "black"
EMPTY = "."
INITIAL_BOARD = (
    "rheakaehr"
    "........."
    ".c.....c."
    "p.p.p.p.p"
    "........."
    "........."
    "P.P.P.P.P"
    ".C.....C."
    "........."
    "RHEAKAEHR"
)

PIECE_NAMES = {
    "K": "帅", "A": "仕", "E": "相", "H": "马", "R": "车", "C": "炮", "P": "兵",
    "k": "将", "a": "士", "e": "象", "h": "马", "r": "车", "c": "炮", "p": "卒",
}
CN_NUMERALS = "零一二三四五六七八九"


@dataclass(frozen=True)
class Move:
    src: int
    dst: int
    captured: str = EMPTY

    @property
    def key(self) -> str:
        return f"{square_name(self.src)}-{square_name(self.dst)}"


def rc(index: int) -> Tuple[int, int]:
    return divmod(index, COLS)


def idx(row: int, col: int) -> int:
    return row * COLS + col


def inside(row: int, col: int) -> bool:
    return 0 <= row < ROWS and 0 <= col < COLS


def square_name(index: int) -> str:
    """返回仅供数据集和开局库使用的内部坐标。"""
    row, col = rc(index)
    return f"{chr(ord('a') + col)}{row}"


def _file_number(col: int, side: str) -> int:
    """中国象棋的路数从行棋方右侧开始计数。"""
    return 9 - col if side == RED else col + 1


def board_point_name(index: int, perspective: str = RED) -> str:
    """在解释注意力时使用的中文棋盘位置。"""
    row, col = rc(index)
    file_text = CN_NUMERALS[_file_number(col, perspective)]
    distance = (9 - row) if perspective == RED else row
    return f"{file_text}路·离{'红' if perspective == RED else '黑'}方底线第 {distance + 1} 格"


def move_notation(board: Sequence[str], move: Move) -> str:
    """将内部位移转为标准的中国象棋文字记谱。

    例如红方中炮开局表达为“炮二平五”，不再对使用者显示 a0-a1
    这类国际象棋风格坐标。
    """
    piece = board[move.src]
    if piece == EMPTY:
        raise ValueError("起点没有棋子")
    side = piece_side(piece)
    source_row, source_col = rc(move.src)
    destination_row, destination_col = rc(move.dst)
    actor = PIECE_NAMES.get(piece, piece)

    same_file = [
        index
        for index, candidate in enumerate(board)
        if candidate == piece and rc(index)[1] == source_col
    ]
    if len(same_file) > 1:
        ordered = sorted(same_file, key=lambda index: rc(index)[0], reverse=side == BLACK)
        position = ordered.index(move.src)
        if len(ordered) == 2:
            prefix = "前" if position == 0 else "后"
        elif position == 0:
            prefix = "前"
        elif position == len(ordered) - 1:
            prefix = "后"
        elif len(ordered) > 3:
            prefix = CN_NUMERALS[position + 1]
        else:
            prefix = "中"
        head = prefix + actor
    else:
        head = actor + CN_NUMERALS[_file_number(source_col, side)]

    if destination_row == source_row:
        action = "平"
        suffix = CN_NUMERALS[_file_number(destination_col, side)]
    else:
        forward = destination_row < source_row if side == RED else destination_row > source_row
        action = "进" if forward else "退"
        if piece.upper() in {"H", "E", "A"}:
            suffix = CN_NUMERALS[_file_number(destination_col, side)]
        else:
            suffix = CN_NUMERALS[abs(destination_row - source_row)]
    return head + action + suffix


def parse_square(name: str) -> int:
    if len(name) != 2 or not ("a" <= name[0].lower() <= "i") or not name[1].isdigit():
        raise ValueError(f"无效坐标：{name}")
    col = ord(name[0].lower()) - ord("a")
    row = int(name[1])
    if not inside(row, col):
        raise ValueError(f"坐标超出棋盘：{name}")
    return idx(row, col)


def parse_move(text: str) -> Tuple[int, int]:
    parts = text.strip().lower().replace("→", "-").split("-")
    if len(parts) != 2:
        raise ValueError("走法格式应为 a6-a5")
    return parse_square(parts[0]), parse_square(parts[1])


def piece_side(piece: str) -> Optional[str]:
    if piece == EMPTY:
        return None
    return RED if piece.isupper() else BLACK


def opponent(side: str) -> str:
    return BLACK if side == RED else RED


def board_to_string(board: Sequence[str]) -> str:
    return "".join(board)


def board_from_string(text: str) -> List[str]:
    compact = text.replace("/", "").strip()
    if len(compact) != ROWS * COLS:
        raise ValueError("棋盘字符串必须恰好包含 90 个格子")
    return list(compact)


def initial_board() -> List[str]:
    return list(INITIAL_BOARD)


def _palace(side: str, row: int, col: int) -> bool:
    if not 3 <= col <= 5:
        return False
    return (7 <= row <= 9) if side == RED else (0 <= row <= 2)


def _add_if_available(board: Sequence[str], side: str, row: int, col: int, out: List[int]) -> None:
    if inside(row, col) and piece_side(board[idx(row, col)]) != side:
        out.append(idx(row, col))


def _piece_destinations(board: Sequence[str], source: int, attacks: bool = False) -> List[int]:
    piece = board[source]
    side = piece_side(piece)
    if side is None:
        return []
    row, col = rc(source)
    kind = piece.upper()
    out: List[int] = []

    if kind == "R":
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r, c = row + dr, col + dc
            while inside(r, c):
                target = board[idx(r, c)]
                if target == EMPTY:
                    out.append(idx(r, c))
                else:
                    if piece_side(target) != side:
                        out.append(idx(r, c))
                    break
                r, c = r + dr, c + dc

    elif kind == "C":
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r, c, screen = row + dr, col + dc, False
            while inside(r, c):
                target = board[idx(r, c)]
                if not screen:
                    if target == EMPTY:
                        if not attacks:
                            out.append(idx(r, c))
                    else:
                        screen = True
                elif target != EMPTY:
                    if piece_side(target) != side:
                        out.append(idx(r, c))
                    break
                r, c = r + dr, c + dc

    elif kind == "H":
        jumps = (
            (-2, -1, -1, 0), (-2, 1, -1, 0), (2, -1, 1, 0), (2, 1, 1, 0),
            (-1, -2, 0, -1), (1, -2, 0, -1), (-1, 2, 0, 1), (1, 2, 0, 1),
        )
        for dr, dc, lr, lc in jumps:
            if inside(row + lr, col + lc) and board[idx(row + lr, col + lc)] == EMPTY:
                _add_if_available(board, side, row + dr, col + dc, out)

    elif kind == "E":
        for dr, dc in ((-2, -2), (-2, 2), (2, -2), (2, 2)):
            r, c = row + dr, col + dc
            eye_r, eye_c = row + dr // 2, col + dc // 2
            own_half = r >= 5 if side == RED else r <= 4
            if inside(r, c) and own_half and board[idx(eye_r, eye_c)] == EMPTY:
                _add_if_available(board, side, r, c, out)

    elif kind == "A":
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            r, c = row + dr, col + dc
            if _palace(side, r, c):
                _add_if_available(board, side, r, c, out)

    elif kind == "K":
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r, c = row + dr, col + dc
            if _palace(side, r, c):
                _add_if_available(board, side, r, c, out)
        # 将帅在同一纵线且中间无子时可互相攻击。
        step = -1 if side == RED else 1
        r = row + step
        while inside(r, col):
            target = board[idx(r, col)]
            if target != EMPTY:
                if target.upper() == "K" and piece_side(target) != side:
                    out.append(idx(r, col))
                break
            r += step

    elif kind == "P":
        forward = -1 if side == RED else 1
        _add_if_available(board, side, row + forward, col, out)
        crossed = row <= 4 if side == RED else row >= 5
        if crossed:
            _add_if_available(board, side, row, col - 1, out)
            _add_if_available(board, side, row, col + 1, out)

    return out


def pseudo_legal_moves(board: Sequence[str], side: str) -> List[Move]:
    moves: List[Move] = []
    for source, piece in enumerate(board):
        if piece_side(piece) == side:
            for destination in _piece_destinations(board, source):
                moves.append(Move(source, destination, board[destination]))
    return moves


def apply_unchecked(board: Sequence[str], move: Move) -> List[str]:
    result = list(board)
    result[move.dst] = result[move.src]
    result[move.src] = EMPTY
    return result


def is_in_check(board: Sequence[str], side: str) -> bool:
    king = "K" if side == RED else "k"
    try:
        king_pos = board.index(king)
    except ValueError:
        return True
    enemy = opponent(side)
    for source, piece in enumerate(board):
        if piece_side(piece) == enemy and king_pos in _piece_destinations(board, source, attacks=True):
            return True
    return False


def legal_moves(board: Sequence[str], side: str) -> List[Move]:
    return [move for move in pseudo_legal_moves(board, side) if not is_in_check(apply_unchecked(board, move), side)]


def apply_move(board: Sequence[str], side: str, source: int, destination: int) -> List[str]:
    for move in legal_moves(board, side):
        if move.src == source and move.dst == destination:
            return apply_unchecked(board, move)
    raise ValueError(f"非法走法：{square_name(source)}-{square_name(destination)}")


def winner(board: Sequence[str], side_to_move: str) -> Optional[str]:
    if "K" not in board:
        return BLACK
    if "k" not in board:
        return RED
    if not legal_moves(board, side_to_move):
        return opponent(side_to_move)
    return None


def move_description(board: Sequence[str], move: Move) -> str:
    capture = f"，吃掉{PIECE_NAMES.get(move.captured, move.captured)}" if move.captured != EMPTY else ""
    return f"{move_notation(board, move)}{capture}"


def render_text(board: Sequence[str]) -> str:
    lines = ["    a  b  c  d  e  f  g  h  i"]
    for row in range(ROWS):
        cells = [PIECE_NAMES.get(board[idx(row, col)], "·") for col in range(COLS)]
        lines.append(f"{row}  " + "  ".join(cells))
    return "\n".join(lines)
