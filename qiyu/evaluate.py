"""在冻结测试集上评估续训模型，并生成可追溯的 JSON/SVG 产物。"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from .encoding import TOKEN_TO_ID, encode_board
from .model import joint_move_log_probs, load_checkpoint
from .retrain import read_jsonl
from .rules import board_from_string, legal_moves


@torch.no_grad()
def run_policy(model, records: List[Dict], device: torch.device, remove_side: bool = False) -> Dict[str, float]:
    legal_top1 = legal_top3 = raw_legal = 0
    reciprocal_rank = 0.0
    latencies: List[float] = []
    for record in records:
        board = board_from_string(record["board"])
        inputs = encode_board(board, record["side"]).unsqueeze(0).to(device)
        if remove_side:
            inputs[0, 0] = TOKEN_TO_ID["<RED>"]
        started = time.perf_counter()
        source_logits, destination_logits, _, _ = model(inputs)
        joint = joint_move_log_probs(source_logits, destination_logits)
        if device.type == "mps":
            torch.mps.synchronize()
        latencies.append((time.perf_counter() - started) * 1000)
        raw_source, raw_destination = divmod(int(joint[0].reshape(-1).argmax().item()), 90)
        legal = legal_moves(board, record["side"])
        legal_pairs = {(move.src, move.dst) for move in legal}
        raw_legal += int((raw_source, raw_destination) in legal_pairs)
        ranked = sorted(
            ((float(joint[0, move.src, move.dst].item()), move.src, move.dst) for move in legal),
            reverse=True,
        )
        target = (record["source"], record["destination"])
        ranked_pairs = [item[1:] for item in ranked]
        legal_top1 += int(ranked_pairs[0] == target)
        legal_top3 += int(target in ranked_pairs[:3])
        if target in ranked_pairs:
            reciprocal_rank += 1 / (ranked_pairs.index(target) + 1)
    count = len(records)
    return {
        "legal_top1_accuracy": legal_top1 / max(count, 1),
        "legal_top3_accuracy": legal_top3 / max(count, 1),
        "legal_mrr": reciprocal_rank / max(count, 1),
        "raw_action_legality": raw_legal / max(count, 1),
        "mean_model_latency_ms": sum(latencies[10:]) / max(len(latencies[10:]), 1),
    }


def make_svg(metrics_csv: Path, output: Path) -> None:
    with metrics_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    width, height, margin = 760, 420, 58
    train = [float(row["train_loss"]) for row in rows]
    validation = [float(row["validation_loss"]) for row in rows]
    ymax = max(train + validation) * 1.08

    def point(index: int, value: float) -> Tuple[float, float]:
        x = margin + index * (width - 2 * margin) / max(len(rows) - 1, 1)
        y = height - margin - value / ymax * (height - 2 * margin)
        return x, y

    def polyline(values: List[float]) -> str:
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(i, value) for i, value in enumerate(values)))

    grid = []
    for index in range(5):
        y = margin + index * (height - 2 * margin) / 4
        label = ymax * (1 - index / 4)
        grid.append(f'<line x1="{margin}" y1="{y:.1f}" x2="{width-margin}" y2="{y:.1f}" stroke="#ddd1ba"/>')
        grid.append(f'<text x="{margin-10}" y="{y+4:.1f}" text-anchor="end" font-size="12">{label:.1f}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="{width}" height="{height}" fill="#fffaf0"/>
<text x="{margin}" y="30" font-family="sans-serif" font-size="19" font-weight="bold" fill="#173f35">棋语 v3 续训曲线</text>
{''.join(grid)}
<polyline points="{polyline(train)}" fill="none" stroke="#a83b2f" stroke-width="3"/>
<polyline points="{polyline(validation)}" fill="none" stroke="#173f35" stroke-width="3"/>
<text x="{margin}" y="{height-15}" font-family="sans-serif" font-size="12">训练损失（红）　验证损失（绿）　续训 1–{len(rows)} 轮</text>
</svg>'''
    output.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="评估棋语续训模型")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/training_v4/best_model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation_v4"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    records = read_jsonl(args.data_file)
    model, extra = load_checkpoint(str(args.checkpoint), device)
    full = run_policy(model, records, device)
    no_side = run_policy(model, records, device, remove_side=True)
    train_positions = {
        record["board"] + "|" + record["side"] for record in read_jsonl(args.train_file)
    }
    unseen_records = [
        record for record in records if record["board"] + "|" + record["side"] not in train_positions
    ]
    unseen = run_policy(model, unseen_records, device)
    results = {
        "evaluation_records": len(records),
        "parameters": model.parameter_count,
        "checkpoint_continuation_epoch": extra.get("metrics", {}).get("continuation_epoch"),
        "full_model_with_rule_filter": full,
        "unseen_position_records": len(unseen_records),
        "unseen_position_metrics": unseen,
        "corrupted_side_token_ablation": no_side,
        "rule_filter_effect": {
            "before_filter_legal_rate": full["raw_action_legality"],
            "after_filter_legal_rate": 1.0,
        },
    }
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_svg(args.checkpoint.parent / "metrics.csv", args.output_dir / "training_curve.svg")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
