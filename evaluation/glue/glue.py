# train_glue_pl.py
# works on container: wpertsch/slurm:0.0.1 + pip install evaluate ToDo
import os
# Avoid tokenizers/datasets fork deadlocks. Is relevant for some tasks/epoch numbers --> num workers on 0 works too!
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_DATASETS_DISABLE_PARALLEL", "1")

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

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


# GLUE task metadata
GLUE_TASKS = {
    # name: (sentence1_key, sentence2_key, num_labels, is_regression)
    "cola":   ("sentence", None, 2, False),
    "sst2":   ("sentence", None, 2, False),
    "mrpc":   ("sentence1", "sentence2", 2, False),
    "qqp":    ("question1", "question2", 2, False),
    "stsb":   ("sentence1", "sentence2", 1, True),
    "mnli":   ("premise", "hypothesis", 3, False),
    "qnli":   ("question", "sentence", 2, False),
    "rte":    ("sentence1", "sentence2", 2, False),
    "wnli":   ("sentence1", "sentence2", 2, False),
}


@dataclass
class GlueConfig:
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
    num_workers: int = 0        # no forking by default
    fp16: bool = False
    trust_remote_code: bool = False  # has to manually set True for ModernBERT or custom repos


# DataModule that encapsulates all data logic like: downloading, tokenization, batching
class GlueDataModule(L.LightningDataModule):
    def __init__(self, cfg: GlueConfig):
        super().__init__()
        self.cfg = cfg
        self.s1, self.s2, _, _ = GLUE_TASKS[cfg.task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, use_fast=True, trust_remote_code=cfg.trust_remote_code
        )
        self.collate = DataCollatorWithPadding(self.tokenizer)
        self.ds = None
        self.has_mnli = cfg.task_name == "mnli"
        self.pin = torch.cuda.is_available()  # only pin on CUDA GPUs --> works on gammaweb

    def setup(self, stage: Optional[str] = None):
        raw = load_dataset("glue", self.cfg.task_name)  # downloads GLUE dataset split

        def tokenize(ex):
            #  defines how to tokenize each example (handles one or two sentences)
            texts = (ex[self.s1],) if self.s2 is None else (ex[self.s1], ex[self.s2])
            return self.tokenizer(*texts, truncation=True, max_length=self.cfg.max_length)

        def rm_cols(split_name: str) -> List[str]:
            #  removes unused columns to keep dataset as small as possible
            cols = raw[split_name].column_names
            keep = {"label"}
            return [c for c in cols if c not in keep]

        mapped = {}
        for split_name in raw.keys():
            mapped[split_name] = raw[split_name].map(
                tokenize, batched=True, remove_columns=rm_cols(split_name)
            )
        self.ds = mapped

    def _dl(self, split, bsz: int, shuffle: bool):
        #  dataloader boiler code
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
        return self._dl(self.ds["train"], self.cfg.batch_size, shuffle=True)

    def val_dataloader(self):
        if self.has_mnli:
            return [
                self._dl(self.ds["validation_matched"], self.cfg.eval_batch_size, shuffle=False),
                self._dl(self.ds["validation_mismatched"], self.cfg.eval_batch_size, shuffle=False),
            ]
        return self._dl(self.ds["validation"], self.cfg.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        if self.has_mnli:
            return [
                self._dl(self.ds["test_matched"], self.cfg.eval_batch_size, shuffle=False),
                self._dl(self.ds["test_mismatched"], self.cfg.eval_batch_size, shuffle=False),
            ]
        return self._dl(self.ds["test"], self.cfg.eval_batch_size, shuffle=False)


# LightningModule
class LightningBERT(L.LightningModule):
    #  encapsulates model, loss, optimizer, scheduler, and metrics
    def __init__(self, cfg: GlueConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        _, _, num_labels, is_reg = GLUE_TASKS[cfg.task_name]
        self.is_regression = is_reg
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            num_labels=num_labels,
            problem_type="regression" if is_reg else "single_label_classification",
            trust_remote_code=cfg.trust_remote_code,
        )
        self.metric = evaluate.load("glue", cfg.task_name)

    def forward(self, **batch):
        return self.model(**batch)

    def _suffix_for_split(self, dataloader_idx: Optional[int]):
        multi = self.cfg.task_name == "mnli"
        return f"_{dataloader_idx}" if (multi and dataloader_idx is not None) else ""

    def common_step(self, batch):
        # Accept both "labels" (collator output) and "label" (dataset field) <-- ToDo this is dirty >:(
        labels = batch.pop("labels", None)
        if labels is None:
            labels = batch.pop("label")
        out = self(**batch, labels=labels)
        return out.loss, out.logits, labels

    def training_step(self, batch, _):
        loss, logits, labels = self.common_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def _compute_and_log_metrics(self, logits, labels, stage: str, dataloader_idx: Optional[int] = None):
        #  dataloader_idx is only needed for tasks with mult. val/test sets (like MNLI)
        if self.is_regression:
            preds = logits.squeeze().detach().cpu().float()
            labels = labels.detach().cpu().float()
        else:
            preds = torch.argmax(logits, dim=-1).detach().cpu()
            labels = labels.detach().cpu()
        metrics: Dict[str, Any] = self.metric.compute(predictions=preds, references=labels)
        tag = self._suffix_for_split(dataloader_idx)
        for k, v in metrics.items():
            self.log(f"{stage}{tag}_{k}", v, prog_bar=True, on_epoch=True, sync_dist=True)
        return metrics

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, logits, labels = self.common_step(batch)
        tag = self._suffix_for_split(dataloader_idx)
        self.log(f"val_loss{tag}", loss, prog_bar=False, on_epoch=True, sync_dist=True)
        self._compute_and_log_metrics(logits, labels, "val", dataloader_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        _, logits, labels = self.common_step(batch)
        self._compute_and_log_metrics(logits, labels, "test", dataloader_idx)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        wd = self.cfg.weight_decay
        grouped = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": wd},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(grouped, lr=self.cfg.lr, betas=(0.9, 0.999), eps=1e-8)

        # Scheduler with warmup
        steps_per_epoch = max(1, self.trainer.estimated_stepping_batches // self.cfg.num_epochs)
        num_training_steps = steps_per_epoch * self.cfg.num_epochs
        num_warmup_steps = int(self.cfg.warmup_ratio * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


# Entrypoint
def main():
    import argparse
    parser = argparse.ArgumentParser()
    # Core training args
    parser.add_argument("--task", type=str, default="sst2", choices=list(GLUE_TASKS.keys()))
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--eval_bsz", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--precision", type=str, default="32-true", choices=["32-true","16-mixed","bf16-mixed"])
    parser.add_argument("--outdir", type=str, default="lightning_logs")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 avoids forking)")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Allow custom model/tokenizer code from HF repos (e.g., ModernBERT)")

    # W&B args
    parser.add_argument("--wandb_project", type=str, default="glue-finetuning")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None)
    parser.add_argument("--wandb_mode", type=str, default=None, choices=[None, "offline", "disabled", "online"])

    # I/O & callbacks
    parser.add_argument("--no_ckpt", action="store_true", help="Disable model checkpointing (useful on slow/NFS storage)")

    # Optional trainer batch limits (handy for smoke tests)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)

    args = parser.parse_args()

    # Repro
    L.seed_everything(args.seed)

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

    # Data & model
    dm = GlueDataModule(cfg)
    model = LightningBERT(cfg)

    # Loggers (CSV + W&B)
    csv_logger = CSVLogger(save_dir=args.outdir, name=f"{args.task}-{args.model.replace('/','_')}")
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_run_name or f"{args.task}-{args.model.replace('/','_')}",
        group=args.wandb_group,
        tags=args.wandb_tags,
        config=vars(args),
        save_dir=args.outdir,
        mode=args.wandb_mode,
    )
    loggers = [csv_logger, wandb_logger]

    # Callbacks
    callbacks = [LearningRateMonitor(logging_interval="step")]
    if not args.no_ckpt:
        callbacks.append(
            ModelCheckpoint(
                monitor="val_loss",     # <— works with Option B naming
                mode="min",
                save_top_k=1,
                save_last=True,
                filename="{epoch}-{step}-{val_loss:.4f}",
            )
        )

    # Trainer
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

    # Fit & test
    trainer.fit(model, dm)

    # Test with best/last if available --> ToDo does not work on cluster
    best_ckpt = None
    if not args.no_ckpt:
        for cb in callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.best_model_path:
                best_ckpt = cb.best_model_path
                break
        if best_ckpt is None:
            ckpt_dir = os.path.join(csv_logger.log_dir, "checkpoints")
            last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
            if os.path.exists(last_ckpt):
                best_ckpt = last_ckpt

    trainer.test(model, datamodule=dm, ckpt_path=best_ckpt)

if __name__ == "__main__":
    main()
