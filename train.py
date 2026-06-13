"""Production-grade trainer for Chinese toxic-speech (binary) classification.

This script fine-tunes a pretrained Chinese transformer encoder
(default: ``hfl/chinese-roberta-wwm-ext``) on the ToxiCN dataset to predict
whether a comment is toxic (1) or non-toxic (0) -- Level I of the Monitor
Toxic Frame taxonomy.

Highlights
----------
* HuggingFace ``AutoModelForSequenceClassification`` backbone (any BERT/RoBERTa
  checkpoint works via ``--model-name``).
* Stratified train/validation split carved out of the official train set; the
  official test set is kept as a held-out evaluation set.
* AdamW + linear warmup/decay schedule, gradient clipping, optional class
  weighting, and automatic mixed precision (AMP) on CUDA.
* Early stopping on validation macro-F1, best-checkpoint saving (model +
  tokenizer + run config), structured logging to console and file, and full
  metric reporting (accuracy / precision / recall / F1 / macro-F1 / confusion
  matrix).
* Reproducible (global seeding) and runnable end-to-end on CPU or GPU.

Examples
--------
Train with defaults (GPU if available)::

    uv run python train.py train

Fast smoke test on a small subset::

    uv run python train.py train --max-samples 256 --epochs 1

Run inference with a trained checkpoint::

    uv run python train.py predict --checkpoint outputs/best --text "你这个废物"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "ToxiCN_ex" / "ToxiCN_ex" / "ToxiCN" / "data"
DEFAULT_TRAIN_JSON = DATA_DIR / "train.json"
DEFAULT_TEST_JSON = DATA_DIR / "test.json"

LABEL_NAMES = ["non-toxic", "toxic"]

logger = logging.getLogger("toxicn")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """All knobs for a training run (mirrors the CLI flags)."""

    model_name: str = "hfl/chinese-roberta-wwm-ext"
    train_json: str = str(DEFAULT_TRAIN_JSON)
    test_json: str = str(DEFAULT_TEST_JSON)
    output_dir: str = str(PROJECT_ROOT / "outputs")

    # data / tokenisation
    max_length: int = 128
    val_ratio: float = 0.1
    max_samples: int = 0  # 0 == use everything (>0 truncates for smoke tests)

    # optimisation
    epochs: int = 3
    batch_size: int = 32
    eval_batch_size: int = 64
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    class_weighting: bool = False

    # runtime
    seed: int = 42
    num_workers: int = 0
    patience: int = 2  # early-stopping patience (epochs); 0 disables
    fp16: bool = True  # AMP on CUDA
    no_cuda: bool = False
    log_every: int = 50

    num_labels: int = field(default=2, init=False)


# ---------------------------------------------------------------------------
# Reproducibility & logging
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    handlers.append(logging.FileHandler(output_dir / "train.log", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    # Silence chatty third-party loggers (HF Hub download HTTP traffic, etc.).
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def resolve_device(no_cuda: bool) -> torch.device:
    if not no_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_records(path: str) -> tuple[list[str], list[int]]:
    """Load (content, toxic-label) pairs from a ToxiCN json file."""
    with open(path, "r", encoding="utf-8") as fh:
        rows = json.load(fh)

    texts: list[str] = []
    labels: list[int] = []
    skipped = 0
    for row in rows:
        content = (row.get("content") or "").strip()
        if not content:
            skipped += 1
            continue
        texts.append(content)
        labels.append(int(row["toxic"]))
    if skipped:
        logger.warning("Skipped %d empty rows from %s", skipped, path)
    return texts, labels


class ToxiDataset(Dataset):
    """Tokenises (lazily-cached) Chinese comments for sequence classification."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        item = {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def make_collate_fn(tokenizer):
    """Dynamic padding to the longest sequence in each batch."""

    def collate(batch: list[dict]) -> dict:
        labels = torch.stack([b.pop("labels") for b in batch])
        padded = tokenizer.pad(batch, padding=True, return_tensors="pt")
        padded["labels"] = labels
        return padded

    return collate


def stratified_split(
    texts: list[str], labels: list[int], val_ratio: float, seed: int
) -> tuple[list[str], list[int], list[str], list[int]]:
    if val_ratio <= 0:
        return texts, labels, [], []
    tr_x, va_x, tr_y, va_y = train_test_split(
        texts, labels, test_size=val_ratio, random_state=seed, stratify=labels
    )
    return tr_x, tr_y, va_x, va_y


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    # Binary metrics computed w.r.t. the positive (toxic == 1) class.
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------------------
# Train / evaluate loops
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, loader, device, loss_fn) -> tuple[dict, float]:
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        n_batches += 1
        all_pred.extend(logits.argmax(dim=-1).cpu().tolist())
        all_true.extend(labels.cpu().tolist())
    metrics = compute_metrics(all_true, all_pred)
    return metrics, total_loss / max(n_batches, 1)


def train(cfg: TrainConfig) -> dict:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    output_dir = Path(cfg.output_dir)
    setup_logging(output_dir)
    set_seed(cfg.seed)
    device = resolve_device(cfg.no_cuda)

    logger.info("Run configuration:\n%s", json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # ---- data ----------------------------------------------------------
    train_texts, train_labels = load_records(cfg.train_json)
    test_texts, test_labels = load_records(cfg.test_json)

    if cfg.max_samples and cfg.max_samples > 0:
        train_texts, train_labels = train_texts[: cfg.max_samples], train_labels[: cfg.max_samples]
        test_texts, test_labels = test_texts[: cfg.max_samples], test_labels[: cfg.max_samples]
        logger.warning("SMOKE TEST: truncated to %d samples per split", cfg.max_samples)

    tr_x, tr_y, va_x, va_y = stratified_split(
        train_texts, train_labels, cfg.val_ratio, cfg.seed
    )
    logger.info(
        "Splits -> train: %d | val: %d | test: %d", len(tr_x), len(va_x), len(test_texts)
    )

    # ---- tokenizer & model --------------------------------------------
    logger.info("Loading tokenizer & model: %s", cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        id2label={i: n for i, n in enumerate(LABEL_NAMES)},
        label2id={n: i for i, n in enumerate(LABEL_NAMES)},
    ).to(device)

    collate = make_collate_fn(tokenizer)
    train_ds = ToxiDataset(tr_x, tr_y, tokenizer, cfg.max_length)
    test_ds = ToxiDataset(test_texts, test_labels, tokenizer, cfg.max_length)
    val_ds = ToxiDataset(va_x, va_y, tokenizer, cfg.max_length) if va_x else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )
    # Fall back to the test set for model selection if val split disabled.
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate,
            pin_memory=(device.type == "cuda"),
        )
        if val_ds is not None
        else test_loader
    )

    # ---- loss (optional class weighting) -------------------------------
    if cfg.class_weighting:
        counts = np.bincount(tr_y, minlength=cfg.num_labels).astype(np.float64)
        weights = counts.sum() / (cfg.num_labels * np.maximum(counts, 1.0))
        class_weights = torch.tensor(weights, dtype=torch.float, device=device)
        logger.info("Class weights: %s", weights.tolist())
    else:
        class_weights = None
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # ---- optimiser & schedule -----------------------------------------
    no_decay = ("bias", "LayerNorm.weight")
    grouped_params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped_params, lr=cfg.lr)

    steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_amp = cfg.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    logger.info(
        "Optimisation: %d total steps (%d warmup), AMP=%s", total_steps, warmup_steps, use_amp
    )

    # ---- training loop -------------------------------------------------
    best_score = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    best_dir = output_dir / "best"
    history: list[dict] = []
    global_step = 0
    train_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        seen = 0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(**batch).logits
                loss = loss_fn(logits, labels) / cfg.grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * cfg.grad_accum_steps
            seen += 1

            if step % cfg.grad_accum_steps == 0:
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if step % cfg.log_every == 0:
                logger.info(
                    "epoch %d | step %d/%d | loss %.4f | lr %.2e",
                    epoch,
                    step,
                    len(train_loader),
                    running_loss / seen,
                    scheduler.get_last_lr()[0],
                )

        val_metrics, val_loss = evaluate(model, val_loader, device, loss_fn)
        epoch_time = time.time() - epoch_start
        logger.info(
            "[epoch %d] train_loss=%.4f val_loss=%.4f acc=%.4f f1=%.4f macro_f1=%.4f (%.1fs)",
            epoch,
            running_loss / max(seen, 1),
            val_loss,
            val_metrics["accuracy"],
            val_metrics["f1"],
            val_metrics["macro_f1"],
            epoch_time,
        )
        history.append({"epoch": epoch, "train_loss": running_loss / max(seen, 1),
                        "val_loss": val_loss, **{k: v for k, v in val_metrics.items()
                                                 if k != "confusion_matrix"}})

        score = val_metrics["macro_f1"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_no_improve = 0
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            (best_dir / "run_config.json").write_text(
                json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info("  -> new best (macro_f1=%.4f) saved to %s", best_score, best_dir)
        else:
            epochs_no_improve += 1
            if cfg.patience and epochs_no_improve >= cfg.patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs).",
                    epoch,
                    epochs_no_improve,
                )
                break

    logger.info(
        "Training finished in %.1fs. Best val macro_f1=%.4f @ epoch %d",
        time.time() - train_start,
        best_score,
        best_epoch,
    )

    # ---- final test evaluation with the best checkpoint ----------------
    if best_dir.exists():
        logger.info("Reloading best checkpoint for test evaluation: %s", best_dir)
        model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    test_metrics, test_loss = evaluate(model, test_loader, device, loss_fn)
    logger.info("=" * 60)
    logger.info("HELD-OUT TEST RESULTS")
    logger.info("  loss       : %.4f", test_loss)
    logger.info("  accuracy   : %.4f", test_metrics["accuracy"])
    logger.info("  precision  : %.4f", test_metrics["precision"])
    logger.info("  recall     : %.4f", test_metrics["recall"])
    logger.info("  f1 (toxic) : %.4f", test_metrics["f1"])
    logger.info("  macro_f1   : %.4f", test_metrics["macro_f1"])
    logger.info("  confusion  : %s  (rows=true [non-toxic, toxic])", test_metrics["confusion_matrix"])
    logger.info("=" * 60)

    summary = {
        "config": asdict(cfg),
        "best_val_macro_f1": best_score,
        "best_epoch": best_epoch,
        "history": history,
        "test_metrics": test_metrics,
        "test_loss": test_loss,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Wrote metrics to %s", output_dir / "metrics.json")
    return summary


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict(checkpoint: str, texts: list[str], max_length: int, no_cuda: bool) -> list[dict]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = resolve_device(no_cuda)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    model.eval()

    results: list[dict] = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text, truncation=True, max_length=max_length, return_tensors="pt"
            ).to(device)
            probs = torch.softmax(model(**enc).logits, dim=-1)[0].cpu()
            pred = int(probs.argmax())
            results.append(
                {
                    "text": text,
                    "label": LABEL_NAMES[pred],
                    "toxic_prob": float(probs[1]),
                }
            )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chinese toxic-speech classifier (ToxiCN).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    defaults = TrainConfig()
    pt = sub.add_parser("train", help="Fine-tune a model.",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pt.add_argument("--model-name", default=defaults.model_name)
    pt.add_argument("--train-json", default=defaults.train_json)
    pt.add_argument("--test-json", default=defaults.test_json)
    pt.add_argument("--output-dir", default=defaults.output_dir)
    pt.add_argument("--max-length", type=int, default=defaults.max_length)
    pt.add_argument("--val-ratio", type=float, default=defaults.val_ratio)
    pt.add_argument("--max-samples", type=int, default=defaults.max_samples,
                    help="Truncate each split to N samples (0 = all). For smoke tests.")
    pt.add_argument("--epochs", type=int, default=defaults.epochs)
    pt.add_argument("--batch-size", type=int, default=defaults.batch_size)
    pt.add_argument("--eval-batch-size", type=int, default=defaults.eval_batch_size)
    pt.add_argument("--lr", type=float, default=defaults.lr)
    pt.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    pt.add_argument("--warmup-ratio", type=float, default=defaults.warmup_ratio)
    pt.add_argument("--grad-clip", type=float, default=defaults.grad_clip)
    pt.add_argument("--grad-accum-steps", type=int, default=defaults.grad_accum_steps)
    pt.add_argument("--class-weighting", action="store_true", default=defaults.class_weighting)
    pt.add_argument("--seed", type=int, default=defaults.seed)
    pt.add_argument("--num-workers", type=int, default=defaults.num_workers)
    pt.add_argument("--patience", type=int, default=defaults.patience)
    pt.add_argument("--no-fp16", dest="fp16", action="store_false", default=defaults.fp16,
                    help="Disable automatic mixed precision.")
    pt.add_argument("--no-cuda", action="store_true", default=defaults.no_cuda)
    pt.add_argument("--log-every", type=int, default=defaults.log_every)

    pp = sub.add_parser("predict", help="Classify text with a trained checkpoint.",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pp.add_argument("--checkpoint", required=True, help="Path to a saved checkpoint dir.")
    pp.add_argument("--text", action="append", default=None,
                    help="Text to classify (repeatable). If omitted, reads stdin lines.")
    pp.add_argument("--max-length", type=int, default=128)
    pp.add_argument("--no-cuda", action="store_true", default=False)
    return parser


def cfg_from_args(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    for key in vars(cfg):
        if hasattr(args, key):
            setattr(cfg, key, getattr(args, key))
    return cfg


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    print(args)
    if args.command == "train":
        cfg = cfg_from_args(args)
        train(cfg)
        return 0
    if args.command == "predict":
        if args.text:
            texts = args.text
        else:
            texts = [line.strip() for line in sys.stdin if line.strip()]
        if not texts:
            print("No input text provided.", file=sys.stderr)
            return 1
        for res in predict(args.checkpoint, texts, args.max_length, args.no_cuda):
            print(f"[{res['label']:>9}] p(toxic)={res['toxic_prob']:.3f}  {res['text']}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
