import abc
from typing import Any, Literal

import datasets
import lightning as L
import numpy as np
import torch
import torchmetrics
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.modeling_outputs import (
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)


class SequenceClassificationCollator:
    """Collator for sequence classification tasks."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        text_column: str | list[str] = "text",
        label_column: str = "labels",
        multi_labels: int | None = None,
        max_seq_length: int | None = 256,
    ):
        self.tokenizer = tokenizer
        if not isinstance(text_column, str | list):
            raise TypeError("One or more text columns have to be specified.")
        self.text_column = text_column
        self.label_column = label_column
        self.multi_labels = multi_labels
        self.max_seq_length = max_seq_length

    def __call__(self, batch: list[dict[str, str | int | float | list]]) -> dict[str, torch.Tensor]:
        """Perform collation operation on a batch of sequence classification examples."""
        if isinstance(self.text_column, str):
            texts = [str(item[self.text_column]) for item in batch]
        else:
            if self.tokenizer.sep_token:
                texts = [f" {self.tokenizer.sep_token} ".join(str(item[c]) for c in self.text_column) for item in batch]
            else:
                texts = [" ".join(str(item[c]) for c in self.text_column) for item in batch]

        out = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        if self.multi_labels is None:
            out["labels"] = torch.LongTensor([item[self.label_column] for item in batch])
        else:
            # Multi-label: BCE loss expects float labels
            labels = np.zeros(shape=(len(batch), self.multi_labels), dtype=np.float32)
            for i, item in enumerate(batch):
                for j in item[self.label_column]:
                    labels[i, j] = 1
            out["labels"] = torch.FloatTensor(labels)
        return {k: v for k, v in out.items() if k in ["input_ids", "attention_mask", "labels"]}


class RegressionCollator(SequenceClassificationCollator):
    """Collator for regression tasks (e.g., STS-B)."""

    def __call__(self, batch: list[dict[str, str | int | float | list]]) -> dict[str, torch.Tensor]:
        """Perform collation operation on a batch of regression examples."""
        if isinstance(self.text_column, str):
            texts = [str(item[self.text_column]) for item in batch]
        else:
            if self.tokenizer.sep_token:
                texts = [f" {self.tokenizer.sep_token} ".join(str(item[c]) for c in self.text_column) for item in batch]
            else:
                texts = [" ".join(str(item[c]) for c in self.text_column) for item in batch]

        out = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        out["labels"] = torch.FloatTensor([item[self.label_column] for item in batch])
        return {k: v for k, v in out.items() if k in ["input_ids", "attention_mask", "labels"]}


class TokenClassificationCollator:
    """Collator for sequence tagging tasks."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        text_column: str = "tokens",
        label_column: str = "tags",
        max_seq_length: int | None = 256,
    ):
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.max_seq_length = max_seq_length

    def __call__(self, batch: list[dict[str, str | int | float | list]]) -> dict[str, torch.Tensor]:
        """Perform collation operation on a batch of sequence tagging examples."""
        texts = [item[self.text_column] for item in batch]
        out = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate([item[self.label_column] for item in batch]):
            word_ids = out.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:  # Set the special tokens to -100.
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        out["labels"] = torch.LongTensor(labels)
        return {k: v for k, v in out.items() if k in ["input_ids", "attention_mask", "labels"]}


class QuestionAnsweringCollator:
    """Collator for QA tasks."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        text_column: tuple[str, str] = ("question", "context"),
        label_column: tuple[str, str] = ("id", "answers"),
        max_seq_length: int | None = 256,
    ):
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.max_seq_length = max_seq_length

    def __call__(
        self, batch: list[dict[str, str | int | float | list | dict[str, list[str | int]]]]
    ) -> dict[str, torch.Tensor]:
        """Perform collation operation on a batch of QA examples."""
        questions = [q[self.text_column[0]].strip() for q in batch]
        contexts = [c[self.text_column[1]] for c in batch]
        ids = [a[self.label_column[0]].strip() for a in batch]
        answers = [a[self.label_column[1]] for a in batch]

        out = self.tokenizer(
            questions,
            contexts,
            max_length=self.max_seq_length,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )
        offset_mapping = out.pop("offset_mapping")

        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer: dict[str, list[str | int]] = answers[i]
            text = answer["text"][0]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(text)
            sequence_ids = out.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        out["start_positions"] = torch.LongTensor(start_positions)
        out["end_positions"] = torch.LongTensor(end_positions)

        res = {k: v for k, v in out.items() if k in ["input_ids", "attention_mask", "start_positions", "end_positions"]}
        res["context"] = contexts
        res["id"] = ids
        res["offset_mapping"] = offset_mapping
        res["answers"] = answers
        return res


def _set_config_num_labels(config: AutoConfig, num_labels: int) -> None:
    """Set the number of labels on a config, handling both num_labels and num_classes attributes."""
    if hasattr(config, "num_labels"):
        config.num_labels = num_labels
    else:
        config.num_classes = num_labels


class TaskModule(abc.ABC, L.LightningModule):
    """Base LightningModule for evaluation tasks."""

    dataset: dict[str, datasets.Dataset]
    model: PreTrainedModel
    task_type: Literal["classification", "tagging", "similarity", "qa"]
    text_column: str | list[str]
    label_column: str | list[str]
    num_classes: int  # How many unique classes
    multi_label: bool = False
    metric: torchmetrics.Metric
    collator: (
        TokenClassificationCollator | SequenceClassificationCollator | QuestionAnsweringCollator | RegressionCollator
    )

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        pretrained_tokenizer_name_or_path: str,
        max_seq_length: int | None = 512,
        learning_rate: float | None = 1e-5,
        weight_decay: float | None = 1e-6,
        train_batch_size: int | None = 128,
        eval_batch_size: int | None = 128,
        num_workers: int | None = 2,
    ):
        super().__init__()
        self.save_hyperparameters()
        if self.task_type == "classification":
            self._init_classification_()
        elif self.task_type == "tagging":
            self._init_tagging_()
        elif self.task_type == "similarity":
            self._init_similarity_()
        elif self.task_type == "qa":
            self._init_qa_()
        else:
            raise ValueError(f"Unknown task type {self.task_type}")

    def _init_classification_(self) -> None:
        config = AutoConfig.from_pretrained(self.hparams.pretrained_model_name_or_path)
        config.problem_type = "single_label_classification" if not self.multi_label else "multi_label_classification"
        _set_config_num_labels(config, self.num_classes)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.hparams.pretrained_model_name_or_path, config=config
        )
        self.collator = SequenceClassificationCollator(
            tokenizer=AutoTokenizer.from_pretrained(self.hparams.pretrained_tokenizer_name_or_path),
            text_column=self.text_column,
            label_column=self.label_column,
            multi_labels=self.num_classes if self.multi_label else None,
            max_seq_length=self.hparams.max_seq_length,
        )

    def _init_tagging_(self) -> None:
        config = AutoConfig.from_pretrained(self.hparams.pretrained_model_name_or_path)
        config.problem_type = "single_label_classification"
        _set_config_num_labels(config, self.num_classes + 1)  # include -100 dummy label
        config.compile_model = False
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.hparams.pretrained_model_name_or_path, config=config
        )
        self.collator = TokenClassificationCollator(
            tokenizer=AutoTokenizer.from_pretrained(self.hparams.pretrained_tokenizer_name_or_path),
            text_column=self.text_column,
            label_column=self.label_column,
            max_seq_length=self.hparams.max_seq_length,
        )

    def _init_similarity_(self) -> None:
        """Initialize regression/similarity task."""
        config = AutoConfig.from_pretrained(self.hparams.pretrained_model_name_or_path)
        config.problem_type = "regression"
        _set_config_num_labels(config, 1)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.hparams.pretrained_model_name_or_path, config=config
        )
        self.collator = RegressionCollator(
            tokenizer=AutoTokenizer.from_pretrained(self.hparams.pretrained_tokenizer_name_or_path),
            text_column=self.text_column,
            label_column=self.label_column,
            max_seq_length=self.hparams.max_seq_length,
        )

    def _init_qa_(self) -> None:
        config = AutoConfig.from_pretrained(self.hparams.pretrained_model_name_or_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.hparams.pretrained_model_name_or_path, config=config
        )
        self.collator = QuestionAnsweringCollator(
            tokenizer=AutoTokenizer.from_pretrained(self.hparams.pretrained_tokenizer_name_or_path),
            max_seq_length=self.hparams.max_seq_length,
        )

    def _postprocess_classification_(
        self, batch: dict[str, torch.Tensor | Any], out: SequenceClassifierOutput
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.multi_label:
            # Multi-label: return logits, torchmetrics applies sigmoid
            return out.logits, batch["labels"]
        metric_task = getattr(self.metric, "task", None)
        if metric_task == "binary":
            # Binary metric: return positive class logits
            return out.logits[:, 1], batch["labels"]
        # Multiclass: return logits, torchmetrics applies argmax
        return out.logits, batch["labels"]

    def _postprocess_tagging_(
        self, batch: dict[str, torch.Tensor | Any], out: TokenClassifierOutput
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = out.logits.view(-1, out.logits.shape[-1])
        labels = batch["labels"].view(-1)
        # Restrict to "real" labels, i.e., non -100
        logits = logits[labels != -100, :-1]
        labels = labels[labels != -100]
        return logits, labels

    def _postprocess_similarity_(
        self, batch: dict[str, torch.Tensor | Any], out: SequenceClassifierOutput
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Postprocess regression/similarity outputs."""
        return out.logits.squeeze(), batch["labels"]

    def _postprocess_qa_(
        self, batch: dict[str, Any], out: QuestionAnsweringModelOutput
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        n_best = 20
        max_answer_length = 30
        predicted_answers = []
        # Reduce batch and output to per-sample dictionaries
        batch = [{k: v[i] for k, v in batch.items()} for i in range(batch["input_ids"].shape[0])]
        logits = [
            {"start_logits": sl, "end_logits": el} for sl, el in zip(out.start_logits, out.end_logits, strict=False)
        ]

        for example, lgts in zip(batch, logits, strict=False):
            idx = example["id"]
            ctx = example["context"]
            start_logits = lgts["start_logits"]
            end_logits = lgts["end_logits"]
            offsets = example["offset_mapping"]
            answers = []
            start_indexes: list[int] = torch.argsort(start_logits, descending=True)[0 : n_best - 1].cpu().tolist()
            end_indexes: list[int] = torch.argsort(end_logits, descending=True)[0 : n_best - 1].cpu().tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    answers.append(
                        {
                            "text": ctx[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logits[start_index] + end_logits[end_index],
                        }
                    )
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append({"id": idx, "prediction_text": best_answer["text"]})
        targets = []
        for example in batch:
            targets.append({"id": example["id"], "answers": example["answers"]})
        return predicted_answers, targets

    @abc.abstractmethod
    def prepare_data(self) -> None:
        """Prepare the dataset objecct needed for the task. To be implemented by task subclasses."""
        raise NotImplementedError

    def configure_optimizers(self) -> "torch.optim.Optimizer":
        """Set up the optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform train step."""
        if self.task_type == "qa":
            # The model signature differs for QA tasks
            output = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                start_positions=batch["start_positions"],
                end_positions=batch["end_positions"],
                output_hidden_states=False,
            )
        else:
            output = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                output_hidden_states=False,
            )
        self.log("loss/train", output.loss, prog_bar=True)
        return output.loss

    def validation_step(self, batch: dict[str, torch.Tensor | Any]) -> None:
        """Perform validation step."""
        out = self.model(batch["input_ids"], batch["attention_mask"])
        if self.task_type == "classification":
            preds, target = self._postprocess_classification_(batch, out)
        elif self.task_type == "tagging":
            preds, target = self._postprocess_tagging_(batch, out)
        elif self.task_type == "similarity":
            preds, target = self._postprocess_similarity_(batch, out)
        elif self.task_type == "qa":
            preds, target = self._postprocess_qa_(batch, out)
        self.metric.update(preds=preds, target=target)

    def on_validation_start(self) -> None:
        """Perform operations a start of validation step."""
        self.metric.to(self.model.device)

    def on_validation_epoch_start(self) -> None:
        """Perform operations at start of validation epoch."""
        self.metric.reset()

    def on_validation_epoch_end(self) -> None:
        """Perform operations at end of validation epoch."""
        m = self.metric.compute()
        if self.task_type == "qa":
            m = m["f1"]
        self.log(f"val/{type(self.metric).__name__}", m, prog_bar=True)

    def test_step(self, batch: dict[str, torch.Tensor]) -> None:
        """Perform test step."""
        out = self.model(batch["input_ids"], batch["attention_mask"])
        if self.task_type == "classification":
            preds, target = self._postprocess_classification_(batch, out)
        elif self.task_type == "tagging":
            preds, target = self._postprocess_tagging_(batch, out)
        elif self.task_type == "similarity":
            preds, target = self._postprocess_similarity_(batch, out)
        elif self.task_type == "qa":
            preds, target = self._postprocess_qa_(batch, out)
        self.metric.update(preds=preds, target=target)

    def on_test_start(self) -> None:
        """Perform operations at start of test step."""
        self.metric.to(self.model.device)

    def on_test_epoch_start(self) -> None:
        """Perform operations at start of test epoch."""
        self.metric.reset()

    def on_test_epoch_end(self) -> None:
        """Perform operations at end of test epoch."""
        m = self.metric.compute()
        if self.task_type == "qa":
            m = m["f1"]
        self.log(f"test/{type(self.metric).__name__}", m, prog_bar=True)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Create train set dataloader."""
        return DataLoader(
            self.dataset.get("train", []),
            collate_fn=self.collator,
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Create validation set dataloader."""
        return DataLoader(
            self.dataset.get("validation", []),
            collate_fn=self.collator,
            batch_size=self.hparams.eval_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Create test set dataloader."""
        return DataLoader(
            self.dataset.get("test", []),
            collate_fn=self.collator,
            batch_size=self.hparams.eval_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
