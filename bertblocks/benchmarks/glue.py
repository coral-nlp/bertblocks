from typing import Literal

import datasets
import torchmetrics

from bertblocks.benchmarks.base import TaskModule


class GLUETaskModule(TaskModule):
    """Base class for GLUE benchmark tasks."""

    task_name: str
    task_group: Literal["acceptability", "sentiment", "paraphrase", "similarity", "nli", "qa_nli"]

    def prepare_data(self) -> None:
        """Obtain the data corresponding to the task."""
        ds = datasets.load_dataset("glue", self.task_name)
        # GLUE test labels are not public, use validation as test
        self.dataset = {
            "train": ds["train"],
            "validation": ds["validation"],
            "test": ds["validation"],
        }


class CoLA(GLUETaskModule):
    """Corpus of Linguistic Acceptability task.

    Task type: classification
    Evaluation metric: Matthews Correlation Coefficient
    """

    task_name = "cola"
    task_group = "acceptability"
    task_type = "classification"
    text_column = "sentence"
    label_column = "label"
    metric = torchmetrics.MatthewsCorrCoef(task="binary")
    num_classes = 2


class SST2(GLUETaskModule):
    """Stanford Sentiment Treebank task.

    Task type: classification
    Evaluation metric: Accuracy
    """

    task_name = "sst2"
    task_group = "sentiment"
    task_type = "classification"
    text_column = "sentence"
    label_column = "label"
    metric = torchmetrics.Accuracy(task="binary")
    num_classes = 2


class MRPC(GLUETaskModule):
    """Microsoft Research Paraphrase Corpus task.

    Task type: classification (sentence pairs)
    Evaluation metric: F1 Score
    """

    task_name = "mrpc"
    task_group = "paraphrase"
    task_type = "classification"
    text_column = ["sentence1", "sentence2"]  # noqa: RUF012
    label_column = "label"
    metric = torchmetrics.F1Score(task="binary")
    num_classes = 2


class QQP(GLUETaskModule):
    """Quora Question Pairs task.

    Task type: classification (sentence pairs)
    Evaluation metric: F1 Score
    """

    task_name = "qqp"
    task_group = "paraphrase"
    task_type = "classification"
    text_column = ["question1", "question2"]  # noqa: RUF012
    label_column = "label"
    metric = torchmetrics.F1Score(task="binary")
    num_classes = 2


class STSB(GLUETaskModule):
    """Semantic Textual Similarity Benchmark task.

    Task type: regression (similarity)
    Evaluation metric: Spearman Correlation
    """

    task_name = "stsb"
    task_group = "similarity"
    task_type = "similarity"
    text_column = ["sentence1", "sentence2"]  # noqa: RUF012
    label_column = "label"
    metric = torchmetrics.SpearmanCorrCoef()
    num_classes = 1


class MNLI(GLUETaskModule):
    """Multi-Genre Natural Language Inference task.

    Task type: classification (3-way)
    Evaluation metric: Accuracy
    Note: Has matched and mismatched validation sets
    """

    task_name = "mnli"
    task_group = "nli"
    task_type = "classification"
    text_column = ["premise", "hypothesis"]  # noqa: RUF012
    label_column = "label"
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=3)
    num_classes = 3


class MNLIMatched(MNLI):
    """MNLI Matched validation subset."""

    def prepare_data(self) -> None:
        """Load MNLI with matched validation set."""
        ds = datasets.load_dataset("glue", "mnli")
        self.dataset = {
            "train": ds["train"],
            "validation": ds["validation_matched"],
            "test": ds["validation_matched"],  # GLUE test labels not public
        }


class MNLIMismatched(MNLI):
    """MNLI Mismatched validation subset."""

    def prepare_data(self) -> None:
        """Load MNLI with mismatched validation set."""
        ds = datasets.load_dataset("glue", "mnli")
        self.dataset = {
            "train": ds["train"],
            "validation": ds["validation_mismatched"],
            "test": ds["validation_mismatched"],
        }


class QNLI(GLUETaskModule):
    """Question Natural Language Inference task.

    Task type: classification
    Evaluation metric: Accuracy
    """

    task_name = "qnli"
    task_group = "qa_nli"
    task_type = "classification"
    text_column = ["question", "sentence"]  # noqa: RUF012
    label_column = "label"
    metric = torchmetrics.Accuracy(task="binary")
    num_classes = 2


class RTE(GLUETaskModule):
    """Recognizing Textual Entailment task.

    Task type: classification
    Evaluation metric: Accuracy
    """

    task_name = "rte"
    task_group = "nli"
    task_type = "classification"
    text_column = ["sentence1", "sentence2"]  # noqa: RUF012
    label_column = "label"
    metric = torchmetrics.Accuracy(task="binary")
    num_classes = 2


class WNLI(GLUETaskModule):
    """Winograd Natural Language Inference task.

    Task type: classification
    Evaluation metric: Accuracy
    """

    task_name = "wnli"
    task_group = "nli"
    task_type = "classification"
    text_column = ["sentence1", "sentence2"]  # noqa: RUF012
    label_column = "label"
    metric = torchmetrics.Accuracy(task="binary")
    num_classes = 2


TASK_MODULES = [
    CoLA,
    SST2,
    MRPC,
    QQP,
    STSB,
    MNLIMatched,
    MNLIMismatched,
    QNLI,
    RTE,
    WNLI,
]
