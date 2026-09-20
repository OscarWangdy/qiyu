"""在现有 v3 checkpoint 上续训，于当前局面的合法着集合内优化联合策略。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .encoding import encode_board
from .model import QiYuTransformer, load_checkpoint, save_checkpoint
from .rules import board_from_string, board_to_string, legal_moves


def read_jsonl(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@lru_cache(maxsize=120_000)
def legal_pairs(board_text: str, side: str) -> torch.Tensor:
    moves = legal_moves(board_from_string(board_text), side)
    return torch.tensor([(move.src, move.dst) for move in moves], dtype=torch.long)


class PositionDataset(Dataset):
    def __init__(self, records: Sequence[Dict], mirror_augmentation: bool):
        self.records = records
        self.mirror_augmentation = mirror_augmentation

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        board = board_from_string(record["board"])
        source, destination = int(record["source"]), int(record["destination"])
        if self.mirror_augmentation and torch.rand(()) < 0.5:
            mirrored = ["."] * 90
            for position, piece in enumerate(board):
                row, column = divmod(position, 9)
                mirrored[row * 9 + (8 - column)] = piece
            board = mirrored
            source_row, source_column = divmod(source, 9)
            destination_row, destination_column = divmod(destination, 9)
            source = source_row * 9 + (8 - source_column)
            destination = destination_row * 9 + (8 - destination_column)
        result = record.get("result", "*")
        if result == "1-0":
            value = 1.0 if record["side"] == "red" else -1.0
        elif result == "0-1":
            value = 1.0 if record["side"] == "black" else -1.0
        else:
            value = 0.0
        board_text = board_to_string(board)
        return (
            encode_board(board, record["side"]),
            source,
            destination,
            value,
            board_text,
            record["side"],
        )


def masked_policy_loss(
    source_logits: torch.Tensor,
    destination_logits: torch.Tensor,
    sources: torch.Tensor,
    destinations: torch.Tensor,
    board_texts: Sequence[str],
    sides: Sequence[str],
) -> torch.Tensor:
    losses = []
    for index, (board_text, side) in enumerate(zip(board_texts, sides)):
        pairs = legal_pairs(board_text, side).to(source_logits.device)
        legal_scores = source_logits[index, pairs[:, 0]] + destination_logits[index, pairs[:, 0], pairs[:, 1]]
        target_score = source_logits[index, sources[index]] + destination_logits[index, sources[index], destinations[index]]
        losses.append(torch.logsumexp(legal_scores, dim=0) - target_score)
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model: QiYuTransformer, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = top1 = top3 = 0
    loss_sum = value_error = 0.0
    for inputs, sources, destinations, values, board_texts, sides in loader:
        inputs = inputs.to(device)
        sources = sources.to(device=device, dtype=torch.long)
        destinations = destinations.to(device=device, dtype=torch.long)
        values = values.to(device=device, dtype=torch.float32)
        source_logits, destination_logits, predicted_values, _ = model(inputs)
        policy_loss = masked_policy_loss(
            source_logits, destination_logits, sources, destinations, board_texts, sides
        )
        batch_size = inputs.size(0)
        loss_sum += (policy_loss + 0.25 * F.mse_loss(predicted_values, values)).item() * batch_size
        value_error += (predicted_values - values).abs().sum().item()
        for index, (board_text, side) in enumerate(zip(board_texts, sides)):
            pairs = legal_pairs(board_text, side).to(device)
            scores = source_logits[index, pairs[:, 0]] + destination_logits[index, pairs[:, 0], pairs[:, 1]]
            count = min(3, len(pairs))
            best = pairs[torch.topk(scores, count).indices]
            target = torch.stack((sources[index], destinations[index]))
            matches = (best == target).all(dim=1)
            top1 += int(matches[0].item())
            top3 += int(matches.any().item())
        total += batch_size
    return {
        "validation_loss": loss_sum / max(total, 1),
        "legal_top1_accuracy": top1 / max(total, 1),
        "legal_top3_accuracy": top3 / max(total, 1),
        "value_mae": value_error / max(total, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="在棋语 v3 基础上进行多轮续训")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/training_v3_long/best_model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/training_v4"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model, base_metadata = load_checkpoint(str(args.checkpoint), device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    train_path = args.data_dir / "train.jsonl"
    validation_path = args.data_dir / "validation.jsonl"
    train_records = read_jsonl(train_path)
    validation_records = read_jsonl(validation_path)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        PositionDataset(train_records, mirror_augmentation=True),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        PositionDataset(validation_records, mirror_augmentation=False),
        batch_size=args.batch_size,
        shuffle=False,
    )

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "seed": args.seed,
        "parameters": model.parameter_count,
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "policy_loss": "sparse_legal_masked_joint",
        "mirror_augmentation": True,
        "continuation_epochs": args.epochs,
        "base_checkpoint": str(args.checkpoint),
        "base_checkpoint_epoch": base_metadata.get("metrics", {}).get("epoch"),
        "data_hashes": {"train": sha256(train_path), "validation": sha256(validation_path)},
    }
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(environment, ensure_ascii=False, indent=2), flush=True)

    started = time.time()
    metrics: List[Dict] = []
    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        progress = (epoch - 1) / max(args.epochs - 1, 1)
        learning_rate = args.minimum_learning_rate + 0.5 * (
            args.learning_rate - args.minimum_learning_rate
        ) * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        running_loss = 0.0
        examples = 0
        for inputs, sources, destinations, values, board_texts, sides in train_loader:
            inputs = inputs.to(device)
            sources = sources.to(device=device, dtype=torch.long)
            destinations = destinations.to(device=device, dtype=torch.long)
            values = values.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            source_logits, destination_logits, predicted_values, _ = model(inputs)
            policy_loss = masked_policy_loss(
                source_logits, destination_logits, sources, destinations, board_texts, sides
            )
            loss = policy_loss + 0.25 * F.mse_loss(predicted_values, values)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            examples += inputs.size(0)

        validation = evaluate(model, validation_loader, device)
        row = {
            "continuation_epoch": epoch,
            "train_loss": running_loss / max(examples, 1),
            **validation,
            "learning_rate": learning_rate,
            "elapsed_seconds": round(time.time() - started, 2),
        }
        metrics.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        metadata = {"metrics": row, "environment": environment, "base_metadata": base_metadata}
        save_checkpoint(str(args.output_dir / "latest_model.pt"), model, metadata)
        if row["legal_top1_accuracy"] > best_accuracy:
            best_accuracy = row["legal_top1_accuracy"]
            save_checkpoint(str(args.output_dir / "best_model.pt"), model, metadata)
        with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(metrics)

    summary = {
        "status": "completed",
        "continuation_epochs": args.epochs,
        "best_legal_top1_accuracy": best_accuracy,
        "total_seconds": round(time.time() - started, 2),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
