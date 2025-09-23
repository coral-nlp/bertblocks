#!/usr/bin/env python3
"""Example script for finetuning transformers on various NLP tasks using BertBlocks training modules.

Example usage:
    # Classification task with IMDB dataset
    python examples/finetune.py --task classification --dataset imdb

    # Token classification with CoNLL-2003 NER
    python examples/finetune.py --task token_classification --dataset conll2003 --dataset_config ner

    # Question answering with SQuAD
    python examples/finetune.py --task question_answering --dataset squad

    # With scheduler and more epochs
    python examples/finetune.py --task classification --epochs 5 --scheduler_type linear --warmup_ratio 0.1
"""

import argparse

import lightning as L

from bertblocks.training.modules import BertBlocksFinetuningDataModule, BertBlocksFinetuningModule


def main():
    """Finetune a transformer model on various tasks."""
    parser = argparse.ArgumentParser(description="Finetune transformers on various NLP tasks")

    # Task arguments
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "token_classification", "question_answering"],
        default="classification",
        help="Task type to finetune for",
    )

    # Model arguments
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="HuggingFace model name or path")
    parser.add_argument(
        "--num_labels",
        type=int,
        default=None,
        help="Number of labels for classification/token classification (auto-detected if not specified)",
    )

    # Data arguments
    parser.add_argument("--dataset", type=str, default="imdb", help="Dataset name from HuggingFace Hub")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset configuration name")
    parser.add_argument("--text_column", type=str, default="text", help="Name of text column in dataset")
    parser.add_argument(
        "--label_column",
        type=str,
        default=None,
        help="Name of label column in dataset (auto-detected based on task if not specified)",
    )
    parser.add_argument(
        "--context_column",
        type=str,
        default="context",
        help="Name of context column in dataset (for question answering)",
    )
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")

    # Compute arguments
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator type (auto, cpu, gpu, tpu)")
    parser.add_argument("--devices", type=str, default="auto", help="Number of devices or device IDs")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Training precision")
    parser.add_argument("--compile_model", action="store_true", help="Compile model with torch.compile")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")

    # Scheduler arguments
    parser.add_argument(
        "--scheduler_type",
        type=str,
        choices=["linear", "cosine", "constant", "polynomial", None],
        default=None,
        help="Learning rate scheduler type",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Warmup ratio (fraction of total steps)")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps (overrides warmup_ratio)")

    args = parser.parse_args()

    # Set task-specific defaults
    if args.label_column is None:
        if args.task == "classification":
            args.label_column = "label"
        elif args.task == "token_classification":
            args.label_column = "labels"  # CoNLL format uses "labels"
        elif args.task == "question_answering":
            args.label_column = "answers"  # SQuAD format uses "answers"

    # Set task-specific text column defaults for QA
    if args.task == "question_answering" and args.text_column == "text":
        args.text_column = "question"  # QA datasets use "question" column

    # Prepare collator kwargs for QA
    collator_kwargs = {}
    if args.task == "question_answering":
        collator_kwargs["context_column"] = args.context_column

    # Initialize data module
    datamodule = BertBlocksFinetuningDataModule(
        task=args.task,
        pretrained_tokenizer_name_or_path=args.model_name,
        dataset_name_or_path=args.dataset,
        dataset_config_name=args.dataset_config,
        max_sequence_length=args.max_length,
        text_column=args.text_column,
        label_column=args.label_column,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_train=True,
        collator_kwargs=collator_kwargs,
    )

    # Initialize model module
    model = BertBlocksFinetuningModule(
        task=args.task,
        pretrained_model_name_or_path=args.model_name,
        num_labels=args.num_labels,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        compile_model=args.compile_model,
        scheduler_type=args.scheduler_type,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
    )
    trainer.fit(model, datamodule)

    if (
        trainer.datamodule is not None
        and hasattr(trainer.datamodule, "test_dataloader")
        and trainer.datamodule.test_dataloader() is not None
    ):
        trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
