"""用相同开局、交换先后手，对比两个棋语权重的实战表现。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .agent import QiYuAgent, SearchConfig
from .retrain import read_jsonl
from .rules import apply_unchecked, board_to_string, initial_board, legal_moves, opponent, winner


PIECE_VALUES = {"R": 9.0, "C": 4.5, "H": 4.0, "E": 2.0, "A": 2.0, "P": 1.2}


def material_balance(board: list[str]) -> float:
    return sum(
        PIECE_VALUES.get(piece.upper(), 0.0) * (1 if piece.isupper() else -1)
        for piece in board if piece != "."
    )


def opening_positions(records: list[dict], count: int, plies: int) -> list[tuple[list[str], str]]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record["game_id"], []).append(record)
    openings = []
    seen = set()
    for game in groups.values():
        board = initial_board()
        side = "red"
        for record in game[:plies]:
            if board_to_string(board) != record["board"] or side != record["side"]:
                break
            chosen = next(
                (move for move in legal_moves(board, side)
                 if (move.src, move.dst) == (record["source"], record["destination"])),
                None,
            )
            if chosen is None:
                break
            board = apply_unchecked(board, chosen)
            side = opponent(side)
        else:
            key = board_to_string(board) + side
            if key not in seen:
                seen.add(key)
                openings.append((board, side))
                if len(openings) == count:
                    break
    if len(openings) < count:
        raise ValueError("测试棋谱中的不同合法开局不足")
    return openings


def play(
    opening: tuple[list[str], str], new: QiYuAgent, old: QiYuAgent,
    new_side: str, ply_limit: int,
) -> dict:
    board, side = list(opening[0]), opening[1]
    repetitions = {board_to_string(board) + side: 1}
    plies = 0
    for _ in range(ply_limit):
        result = winner(board, side)
        if result is not None:
            break
        agent = new if side == new_side else old
        decision = agent.decide(board, side)
        board = apply_unchecked(board, decision.move)
        side = opponent(side)
        plies += 1
        key = board_to_string(board) + side
        repetitions[key] = repetitions.get(key, 0) + 1
        if repetitions[key] >= 3:
            break
    result = winner(board, side)
    return {
        "new_side": new_side,
        "plies": plies,
        "winner": result,
        "material_for_new": round(material_balance(board) * (1 if new_side == "red" else -1), 2),
        "threefold_repetition": repetitions.get(board_to_string(board) + side, 0) >= 3,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="棋语权重交换先后手对局测试")
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--openings", type=int, default=3)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--ply-limit", type=int, default=60)
    parser.add_argument("--time-limit-ms", type=int, default=200)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = SearchConfig(time_limit_ms=args.time_limit_ms)
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    new = QiYuAgent(args.new, device=device, search_config=config)
    old = QiYuAgent(args.old, device=device, search_config=config)
    openings = opening_positions(read_jsonl(args.test_file), args.openings, args.opening_plies)
    games = []
    for index, opening in enumerate(openings):
        for new_side in ("red", "black"):
            game = play(opening, new, old, new_side, args.ply_limit)
            game["opening"] = index + 1
            games.append(game)
            print(json.dumps(game, ensure_ascii=False), flush=True)
    summary = {
        "new_checkpoint": str(args.new), "old_checkpoint": str(args.old),
        "time_limit_ms": args.time_limit_ms, "opening_plies": args.opening_plies,
        "ply_limit": args.ply_limit, "games": games,
        "new_wins": sum(game["winner"] == game["new_side"] for game in games),
        "old_wins": sum(game["winner"] == opponent(game["new_side"]) for game in games),
        "draws_or_capped": sum(game["winner"] is None for game in games),
        "mean_material_for_new": sum(game["material_for_new"] for game in games) / len(games),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
