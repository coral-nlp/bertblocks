"""
GLUE Fine-tuning with PyTorch Lightning
Evaluates on validation splits (never uses test labels)
"""
import os
import csv
import json
import fcntl
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from collections import defaultdict

# Safe defaults for HF tokenizers/datasets + CPU clusters
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_DATASETS_DISABLE_PARALLEL", "1")

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

# -------------------------
# GLUE task metadata
# -------------------------
GLUE_TASKS = {
    # name: (sentence1_key, sentence2_key, num_labels, is_regression, primary_metric)
    "cola": ("sentence", None, 2, False, "matthews_correlation"),
    "sst2": ("sentence", None, 2, False, "accuracy"),
    "mrpc": ("sentence1", "sentence2", 2, False, "f1"),
    "qqp": ("question1", "question2", 2, False, "f1"),
    "stsb": ("sentence1", "sentence2", 1, True, "spearmanr"),
    "mnli": ("premise", "hypothesis", 3, False, "accuracy"),
    "qnli": ("question", "sentence", 2, False, "accuracy"),
    "rte": ("sentence1", "sentence2", 2, False, "accuracy"),
    "wnli": ("sentence1", "sentence2", 2, False, "accuracy"),
}


@dataclass
class GlueConfig:
    """Configuration for GLUE fine-tuning"""
    model_name: str = "bert-base-uncased"
    task_name: str = "sst2"
    max_length: int = 256
    batch_size: int = 32
    eval_batch_size: int = 64
    lr: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_ratio: float = 0.06
    seed: int = 42
    num_workers: int = 0
    fp16: bool = False
    trust_remote_code: bool = False


# -------------------------
# DataModule
# -------------------------
class GlueDataModule(L.LightningDataModule):
    """DataModule for GLUE tasks - uses validation splits for all evaluation"""

    def __init__(self, cfg: GlueConfig):
        super().__init__()
        self.cfg = cfg
        self.s1, self.s2, _, _, _ = GLUE_TASKS[cfg.task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            use_fast=True,
            trust_remote_code=cfg.trust_remote_code
        )
        self.collate = DataCollatorWithPadding(self.tokenizer)
        self.ds = None
        self.is_mnli = cfg.task_name == "mnli"
        self.pin = torch.cuda.is_available()

    def setup(self, stage: Optional[str] = None):
        """Load and tokenize dataset"""
        raw = load_dataset("glue", self.cfg.task_name)

        def tokenize(ex):
            texts = (ex[self.s1],) if self.s2 is None else (ex[self.s1], ex[self.s2])
            return self.tokenizer(
                *texts,
                truncation=True,
                max_length=self.cfg.max_length
            )

        def get_cols_to_remove(split_name: str) -> List[str]:
            cols = raw[split_name].column_names
            keep = {"label"}
            return [c for c in cols if c not in keep]

        self.ds = {
            s: raw[s].map(
                tokenize,
                batched=True,
                remove_columns=get_cols_to_remove(s)
            )
            for s in raw.keys()
        }

    def _make_dataloader(self, split, bsz: int, shuffle: bool):
        """Create a DataLoader with appropriate settings"""
        return DataLoader(
            split,
            batch_size=bsz,
            shuffle=shuffle,
            collate_fn=self.collate,
            num_workers=self.cfg.num_workers,
            pin_memory=self.pin,
            persistent_workers=(self.cfg.num_workers > 0),
        )

    def train_dataloader(self):
        return self._make_dataloader(self.ds["train"], self.cfg.batch_size, shuffle=True)

    def val_dataloader(self):
        """Returns validation split(s) for monitoring during training"""
        if self.is_mnli:
            return [
                self._make_dataloader(self.ds["validation_matched"], self.cfg.eval_batch_size, False),
                self._make_dataloader(self.ds["validation_mismatched"], self.cfg.eval_batch_size, False),
            ]
        return self._make_dataloader(self.ds["validation"], self.cfg.eval_batch_size, False)

    def test_dataloader(self):
        """Returns validation split(s) for final evaluation (NOT test splits)"""
        return self.val_dataloader()


# -------------------------
# LightningModule
# -------------------------
class GlueModule(L.LightningModule):
    """Lightning module for GLUE fine-tuning"""

    def __init__(self, cfg: GlueConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # Task metadata
        _, _, num_labels, is_reg, self.primary_metric = GLUE_TASKS[cfg.task_name]
        self.is_regression = is_reg

        # Model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            num_labels=num_labels,
            problem_type="regression" if is_reg else "single_label_classification",
            trust_remote_code=cfg.trust_remote_code,
        )

        # Metrics
        self.metric = evaluate.load("glue", cfg.task_name)

        # Accumulators for full-dataset evaluation
        self._eval_preds: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self._eval_labels: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.final_metrics: Dict[str, Dict[str, float]] = {}

    def forward(self, **batch):
        return self.model(**batch)

    def _get_split_suffix(self, dataloader_idx: Optional[int]) -> str:
        """Get suffix for multi-dataloader scenarios (MNLI)"""
        if self.cfg.task_name == "mnli" and dataloader_idx is not None:
            return f"_{dataloader_idx}"
        return ""

    def _common_step(self, batch):
        """Shared forward pass logic"""
        labels = batch.get("labels", batch.get("label"))
        inputs = {k: v for k, v in batch.items() if k not in ("label", "labels")}
        outputs = self(**inputs, labels=labels)
        return outputs.loss, outputs.logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation during training for checkpointing/monitoring"""
        loss, logits, labels = self._common_step(batch)
        suffix = self._get_split_suffix(dataloader_idx)

        # Log loss
        self.log(f"val_loss{suffix}", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        # Compute and log metrics
        if self.is_regression:
            preds = logits.squeeze().detach().cpu().float()
            refs = labels.detach().cpu().float()
        else:
            preds = torch.argmax(logits, dim=-1).detach().cpu()
            refs = labels.detach().cpu()

        metrics = self.metric.compute(predictions=preds, references=refs)
        for metric_name, value in metrics.items():
            self.log(
                f"val{suffix}_{metric_name}",
                value,
                prog_bar=(metric_name == self.primary_metric),
                on_epoch=True,
                sync_dist=True
            )

    def on_test_epoch_start(self):
        """Initialize accumulators for final evaluation"""
        self._eval_preds.clear()
        self._eval_labels.clear()
        self.final_metrics.clear()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """Final evaluation on validation splits (accumulate predictions)"""
        labels = batch.get("labels", batch.get("label"))
        inputs = {k: v for k, v in batch.items() if k not in ("label", "labels")}

        with torch.no_grad():
            outputs = self(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

        # Accumulate predictions and labels
        if self.is_regression:
            self._eval_preds[dataloader_idx].append(logits.squeeze().detach().cpu())
        else:
            self._eval_preds[dataloader_idx].append(
                torch.argmax(logits, dim=-1).detach().cpu()
            )
        self._eval_labels[dataloader_idx].append(labels.detach().cpu())

    def on_test_epoch_end(self):
        """Compute final metrics across entire validation split(s)"""
        for idx in sorted(self._eval_preds.keys()):
            # Concatenate all batches
            preds = torch.cat(self._eval_preds[idx]).numpy()
            refs = torch.cat(self._eval_labels[idx]).numpy()

            # Compute metrics
            metrics = evaluate.load("glue", self.cfg.task_name).compute(
                predictions=preds.tolist(),
                references=refs.tolist()
            )

            # Log metrics
            suffix = self._get_split_suffix(idx)
            split_name = f"validation{suffix}"

            for metric_name, value in metrics.items():
                self.log(
                    f"{split_name}_{metric_name}",
                    value,
                    prog_bar=(metric_name == self.primary_metric),
                    on_epoch=True,
                    sync_dist=True
                )

            # Store for CSV output
            self.final_metrics[split_name] = metrics

            # Print summary
            primary_val = metrics.get(self.primary_metric, list(metrics.values())[0])
            print(f"\n{split_name} {self.primary_metric}: {primary_val:.4f}")

    def configure_optimizers(self):
        """Setup optimizer and learning rate scheduler"""
        # Separate parameters with/without weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.cfg.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Linear warmup + decay schedule
        steps_per_epoch = max(
            1,
            self.trainer.estimated_stepping_batches // self.cfg.num_epochs
        )
        num_training_steps = steps_per_epoch * self.cfg.num_epochs
        num_warmup_steps = int(self.cfg.warmup_ratio * num_training_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps,
            num_training_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


# -------------------------
# CSV Writing with File Locking
# -------------------------
def write_results_to_csv(
        csv_path: Path,
        rows: List[Dict[str, Any]],
):
    """Safely append results to CSV with file locking"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine if we need to write header
    write_header = not csv_path.exists()

    # Collect all field names
    if write_header:
        fieldnames = sorted(set().union(*[row.keys() for row in rows]))
    else:
        # Read existing fieldnames
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames or []

        # Merge with new fields
        all_fields = set(existing_fields) | set().union(*[row.keys() for row in rows])
        fieldnames = list(existing_fields) + [
            f for f in sorted(all_fields) if f not in existing_fields
        ]

        # If new fields were added, we need to rewrite the entire file
        if set(fieldnames) != set(existing_fields):
            with csv_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)

            # Rewrite with new header
            with csv_path.open("w", newline="") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in existing_rows:
                        for field in fieldnames:
                            row.setdefault(field, "")
                        writer.writerow(row)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            write_header = False

    # Append new rows with locking
    with csv_path.open("a", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            for row in rows:
                for field in fieldnames:
                    row.setdefault(field, "")
                writer.writerow(row)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# -------------------------
# Main
# -------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune transformers on GLUE tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core training arguments
    parser.add_argument("--task", type=str, default="sst2",
                        choices=list(GLUE_TASKS.keys()),
                        help="GLUE task name")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="HuggingFace model name")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--bsz", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--eval_bsz", type=int, default=64,
                        help="Evaluation batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max_len", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--accum", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--precision", type=str, default="32-true",
                        choices=["32-true", "16-mixed", "bf16-mixed"],
                        help="Training precision")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (0 avoids forking)")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Trust remote code from model repos")

    # Logging arguments
    parser.add_argument("--outdir", type=str, default="lightning_logs",
                        help="Output directory for logs")
    parser.add_argument("--wandb_project", type=str, default="glue-finetuning",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases run name")
    parser.add_argument("--wandb_group", type=str, default=None,
                        help="Weights & Biases group name")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None,
                        help="Weights & Biases tags")
    parser.add_argument("--wandb_mode", type=str, default="disabled",
                        choices=["online", "offline", "disabled"],
                        help="Weights & Biases mode")

    # Output arguments
    parser.add_argument("--no_ckpt", action="store_true",
                        help="Disable checkpointing")
    parser.add_argument("--results_csv", type=str, default=None,
                        help="CSV file to append results")
    parser.add_argument("--echo_json", action="store_true",
                        help="Print JSON results to stdout")

    # Debug/testing arguments
    parser.add_argument("--limit_train_batches", type=float, default=1.0,
                        help="Limit training batches (for testing)")
    parser.add_argument("--limit_val_batches", type=float, default=1.0,
                        help="Limit validation batches (for testing)")

    args = parser.parse_args()

    # Set seed for reproducibility
    L.seed_everything(args.seed)

    # Create config
    cfg = GlueConfig(
        model_name=args.model,
        task_name=args.task,
        max_length=args.max_len,
        batch_size=args.bsz,
        eval_batch_size=args.eval_bsz,
        lr=args.lr,
        weight_decay=args.wd,
        num_epochs=args.epochs,
        seed=args.seed,
        num_workers=args.num_workers,
        fp16=(args.precision == "16-mixed"),
        trust_remote_code=args.trust_remote_code,
    )

    # Initialize data and model
    dm = GlueDataModule(cfg)
    model = GlueModule(cfg)

    # Setup loggers
    csv_logger = CSVLogger(
        save_dir=args.outdir,
        name=f"{args.task}-{args.model.replace('/', '_')}"
    )

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_run_name or f"{args.task}-{args.model.replace('/', '_')}",
        group=args.wandb_group,
        tags=args.wandb_tags,
        config=vars(args),
        save_dir=args.outdir,
        mode=args.wandb_mode,
    )

    loggers = [csv_logger, wandb_logger]

    # Setup callbacks
    callbacks = [LearningRateMonitor(logging_interval="step")]

    if not args.no_ckpt:
        callbacks.append(
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=True,
                filename="{epoch}-{step}-{val_loss:.4f}",
            )
        )

    # Create trainer
    trainer = L.Trainer(
        max_epochs=cfg.num_epochs,
        precision=args.precision,
        accumulate_grad_batches=args.accum,
        gradient_clip_val=1.0,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_checkpointing=(not args.no_ckpt),
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )

    # Train
    print(f"\n{'=' * 60}")
    print(f"Training {args.model} on {args.task.upper()}")
    print(f"{'=' * 60}\n")

    trainer.fit(model, dm)

    # Get best checkpoint
    best_ckpt = None
    if not args.no_ckpt:
        for cb in callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.best_model_path:
                best_ckpt = cb.best_model_path
                break

        if best_ckpt is None:
            ckpt_dir = Path(csv_logger.log_dir) / "checkpoints"
            last_ckpt = ckpt_dir / "last.ckpt"
            if last_ckpt.exists():
                best_ckpt = str(last_ckpt)

    # Final evaluation on validation splits
    print(f"\n{'=' * 60}")
    print(f"Final Evaluation on Validation Split(s)")
    print(f"{'=' * 60}\n")

    trainer.test(model, datamodule=dm, ckpt_path=best_ckpt)

    # Prepare results
    rows = []
    for split_name, metrics in model.final_metrics.items():
        row = {
            "task": args.task,
            "model": args.model,
            "split": split_name,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.bsz,
            "lr": args.lr,
            "primary_metric": GLUE_TASKS[args.task][4],
            **metrics,
        }
        rows.append(row)

    # Write to CSV
    if args.results_csv:
        csv_path = Path(args.results_csv)
        write_results_to_csv(csv_path, rows)
        print(f"\nResults appended to: {csv_path}")

    # Echo JSON
    if args.echo_json:
        output = {
            "task": args.task,
            "model": args.model,
            "results": model.final_metrics,
        }
        print(json.dumps(output), flush=True)

    print(f"\n{'=' * 60}")
    print(f"Training Complete!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()