from typing import Literal

import datasets
import torchmetrics

from bertblocks.benchmarks.base import TaskModule


class SuperGLEBerTaskModule(TaskModule):
    """Base class for superGLEBER tasks."""

    task_name: str
    task_group: Literal["other", "wsd", "match", "sentiment", "toxicity", "ner"]

    def prepare_data(self) -> None:
        """Obtain the data corresponding to the task."""
        self.dataset = datasets.load_dataset("lgienapp/superGLEBer", self.task_name)


class ArgumentMining(SuperGLEBerTaskModule):
    """Argument Mining task.

    Task type: classification (multilabel)
    Evaluation metric: Macro F1 Score
    """

    task_name = "argument_mining"
    task_group = "other"
    task_type = "classification"
    text_column = "text"
    label_column = "labels"
    metric = torchmetrics.F1Score(task="multilabel", average="macro", num_labels=3, num_classes=3)
    num_classes = 3
    multi_label = True


class DBAspect(SuperGLEBerTaskModule):
    """DB Aspect-based Sentiment Analysis task.

    Task type: classification
    Evaluation metric: Micro F1 Score
    """

    task_name = "db_aspect"
    task_group = "sentiment"
    task_type = "classification"
    text_column = "text"
    label_column = "labels"
    metric = torchmetrics.F1Score(task="multiclass", average="micro", num_classes=60)
    num_classes = 60


class EngagingComments(SuperGLEBerTaskModule):
    """Engaging Comments Detection task.

    Task type: classification
    Evaluation metric: Macro F1 Score
    """

    task_name = "engaging_comments"
    task_group = "other"
    task_type = "classification"
    text_column = "text"
    label_column = "labels"
    metric = torchmetrics.F1Score(task="multiclass", average="macro", num_classes=2)
    num_classes = 2


class FactclaimingComments(SuperGLEBerTaskModule):
    """Factclaiming Comments Detection task.

    Task type: classification
    Evaluation metric: Macro F1 Score
    """

    task_name = "factclaiming_comments"
    task_group = "other"
    task_type = "classification"
    text_column = "text"
    label_column = "labels"
    metric = torchmetrics.F1Score(task="multiclass", average="macro", num_classes=2)
    num_classes = 2


class GermanQuad(SuperGLEBerTaskModule):
    """GermanQuad Question Answering task.

    Task type: qa (question answering)
    Evaluation metric: SQuAD Score
    """

    task_name = "germanquad"
    task_group = "other"
    task_type = "qa"
    text_column = ["question", "context"]  # noqa: RUF012
    label_column = ["id", "answers"]  # noqa: RUF012
    metric = torchmetrics.SQuAD()


class GermevalOpinions(SuperGLEBerTaskModule):
    """GermEval Opinions Aspect-based Sentiment task.

    Task type: tagging (sequence labeling)
    Evaluation metric: Micro F1 Score
    """

    task_name = "germeval_opinions"
    task_group = "other"
    task_type = "tagging"
    text_column = "tokens"
    label_column = "tags"
    metric = torchmetrics.F1Score(task="multiclass", average="micro", num_classes=4)
    num_classes = 4


class HotelAspect(SuperGLEBerTaskModule):
    """Hotel Aspect-based Sentiment Analysis task.

    Task type: classification
    Evaluation metric: Micro F1 Score
    """

    task_name = "hotel_aspect"
    task_group = "other"
    task_type = "classification"
    text_column = "text"
    label_column = "labels"
    metric = torchmetrics.F1Score(task="multiclass", average="micro", num_classes=15)
    num_classes = 15


class MassiveIntents(SuperGLEBerTaskModule):
    """MASSIVE Intent Classification task.

    Task type: classification
    Evaluation metric: Micro F1 Score
    """

    task_name = "massive_intents"
    task_group = "other"
    task_type = "classification"
    text_column = "text"
    label_column = "labels"
    metric = torchmetrics.F1Score(task="multiclass", average="micro", num_classes=60)
    num_classes = 60


class MassiveSeq(SuperGLEBerTaskModule):
    """MASSIVE Sequence Labeling task.

    Task type: tagging (sequence labeling)
    Evaluation metric: Micro F1 Score
    """

    task_name = "massive_seq"
    task_group = "other"
    task_type = "tagging"
    text_column = "tokens"
    label_column = "tags"
    metric = torchmetrics.F1Score(task="multiclass", average="micro", num_classes=56)
    num_classes = 56


class MLQA(SuperGLEBerTaskModule):
    """MLQA Multilingual Question Answering task.

    Task type: qa (question answering)
    Evaluation metric: SQuAD Score
    """

    task_name = "mlqa"
    task_group = "other"
    task_type = "qa"
    text_column = ["question", "context"]  # noqa: RUF012
    label_column = ["id", "answers"]  # noqa: RUF012
    metric = torchmetrics.SQuAD()


class NERBiofid(SuperGLEBerTaskModule):
    """NER Biofid Named Entity Recognition task.

    Task type: tagging (sequence labeling)
    Evaluation metric: Micro F1 Score
    """

    task_name = "ner_biofid"
    task_group = "ner"
    task_type = "tagging"
    text_column = "tokens"
    label_column = "tags"
    metric = torchmetrics.F1Score(task="multiclass", average="micro", num_classes=13)
    num_classes = 13


class NEREuroparl(SuperGLEBerTaskModule):
    """NER Europarl Named Entity Recognition task.

    Task type: tagging (sequence labeling)
    Evaluation metric: Micro F1 Score
    """

    task_name = "ner_europarl"
    task_group = "ner"
    task_type = "tagging"
    text_column = "tokens"
    label_column = "tags"
    metric = torchmetrics.F1Score(task="multiclass", average="micro", num_classes=5)
    num_classes = 5


class NERLegal(SuperGLEBerTaskModule):
    """NER Legal Named Entity Recognition task.

    Task type: tagging (sequence labeling)
    Evaluation metric: Micro F1 Score
    """

    task_name = "ner_legal"
    task_group = "ner"
    task_type = "tagging"
    text_column = "tokens"
    label_column = "tags"
    metric = torchmetrics.F1Score(task="multiclass", average="micro", num_classes=39)
    num_classes = 39


class NERWikiNews(SuperGLEBerTaskModule):
    """NER Wiki News Named Entity Recognition task.

    Task type: tagging (sequence labeling)
    Evaluation metric: Micro F1 Score
    """

    task_name = "ner_wiki_news"
    task_group = "ner"
    task_type = "tagging"
    text_column = "tokens"
    label_column = "tags"
    metric = torchmetrics.F1Score(task="multiclass", average="micro", num_classes=25)
    num_classes = 25


class NERNews(SuperGLEBerTaskModule):
    """NER News Named Entity Recognition task.

    Task type: tagging (sequence labeling)
    Evaluation metric: Micro F1 Score
    """

    task_name = "ner_news"
    task_group = "ner"
    task_type = "tagging"
    text_column = "tokens"
    label_column = "tags"
    metric = torchmetrics.F1Score(task="multiclass", average="micro", num_classes=8)
    num_classes = 8


class NewsClass(SuperGLEBerTaskModule):
    """News Classification task.

    Task type: classification
    Evaluation metric: Accuracy
    """

    task_name = "news_class"
    task_group = "other"
    task_type = "classification"
    text_column = ["title", "text"]  # noqa: RUF012
    label_column = "labels"
    metric = torchmetrics.Accuracy("multiclass", num_classes=10)
    num_classes = 10


class NLI(SuperGLEBerTaskModule):
    """Natural Language Inference task.

    Task type: classification
    Evaluation metric: Accuracy
    """

    task_name = "nli"
    task_group = "other"
    task_type = "classification"
    text_column = ["sentence_1", "sentence_2"]  # noqa: RUF012
    label_column = "labels"
    metric = torchmetrics.Accuracy("multiclass", num_classes=3)
    num_classes = 3


class OffensiveLanguage(SuperGLEBerTaskModule):
    """Offensive Language Detection task.

    Task type: classification
    Evaluation metric: Macro F1 Score
    """

    task_name = "offensive_lang"
    task_group = "toxicity"
    task_type = "classification"
    text_column = "text"
    label_column = "labels"
    metric = torchmetrics.F1Score(task="multiclass", average="macro", num_classes=4)
    num_classes = 4


class ParaphraseMatching(SuperGLEBerTaskModule):
    """Paraphrase Matching task.

    Task type: classification
    Evaluation metric: Accuracy
    """

    task_name = "paraphrase_matching"
    task_group = "match"
    task_type = "classification"
    text_column = ["sentence_1", "sentence_2"]  # noqa: RUF012
    label_column = "labels"
    metric = torchmetrics.Accuracy("multiclass", num_classes=2)
    num_classes = 2


class PAWSX(SuperGLEBerTaskModule):
    """PAWS-X Paraphrase Adversaries from Word Scrambling task.

    Task type: similarity
    Evaluation metric: Pearson Correlation Coefficient
    """

    task_name = "pawsx"
    task_group = "other"
    task_type = "similarity"
    text_column = ["sentence_1", "sentence_2"]  # noqa: RUF012
    label_column = "labels"
    metric = torchmetrics.PearsonCorrCoef()
    num_classes = 0


class Polarity(SuperGLEBerTaskModule):
    """Polarity Sentiment Analysis task.

    Task type: classification
    Evaluation metric: Micro F1 Score
    """

    task_name = "polarity"
    task_group = "sentiment"
    task_type = "classification"
    text_column = "text"
    label_column = "labels"
    metric = torchmetrics.F1Score(task="multiclass", average="micro", num_classes=3)
    num_classes = 3


class QueryAd(SuperGLEBerTaskModule):
    """Query-Ad Matching task.

    Task type: classification
    Evaluation metric: Accuracy
    """

    task_name = "query_ad"
    task_group = "match"
    task_type = "classification"
    text_column = ["sentence_1", "sentence_2", "sentence_3"]  # noqa: RUF012
    label_column = "labels"
    metric = torchmetrics.Accuracy("multiclass", num_classes=2)
    num_classes = 2


class QuestAns(SuperGLEBerTaskModule):
    """Question Answering Matching task.

    Task type: classification
    Evaluation metric: Accuracy
    """

    task_name = "quest_ans"
    task_group = "match"
    task_type = "classification"
    text_column = ["sentence_1", "sentence_2", "sentence_3"]  # noqa: RUF012
    label_column = "labels"
    metric = torchmetrics.Accuracy("multiclass", num_classes=2)
    num_classes = 2


class TopicRelevance(SuperGLEBerTaskModule):
    """Topic Relevance Classification task.

    Task type: classification
    Evaluation metric: Micro F1 Score
    """

    task_name = "topic_relevance"
    task_group = "other"
    task_type = "classification"
    text_column = "text"
    label_column = "labels"
    metric = torchmetrics.F1Score(task="multiclass", average="micro", num_classes=2)
    num_classes = 2


class ToxicComments(SuperGLEBerTaskModule):
    """Toxic Comments Detection task.

    Task type: classification
    Evaluation metric: Macro F1 Score
    """

    task_name = "toxic_comments"
    task_group = "toxicity"
    task_type = "classification"
    text_column = "text"
    label_column = "labels"
    metric = torchmetrics.F1Score(task="multiclass", num_classes=2, average="macro")
    num_classes = 2


class UPDep(SuperGLEBerTaskModule):
    """Universal Dependencies Dependency Parsing task.

    Task type: tagging (sequence labeling)
    Evaluation metric: Micro F1 Score
    """

    task_name = "up_dep"
    task_group = "other"
    task_type = "tagging"
    text_column = "tokens"
    label_column = "tags"
    metric = torchmetrics.F1Score(task="multiclass", average="micro", num_classes=36)
    num_classes = 36


class UPPos(SuperGLEBerTaskModule):
    """Universal Dependencies Part-of-Speech Tagging task.

    Task type: tagging (sequence labeling)
    Evaluation metric: Micro F1 Score
    """

    task_name = "up_pos"
    task_group = "other"
    task_type = "tagging"
    text_column = "tokens"
    label_column = "tags"
    metric = torchmetrics.F1Score(task="multiclass", average="micro", num_classes=16)
    num_classes = 16


class VerbalIdioms(SuperGLEBerTaskModule):
    """Verbal Idioms Word Sense Disambiguation task.

    Task type: classification
    Evaluation metric: Accuracy
    """

    task_name = "verbal_idioms"
    task_group = "wsd"
    task_type = "classification"
    text_column = ["sentence_1", "sentence_2"]  # noqa: RUF012
    label_column = "labels"
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=4)
    num_classes = 4


class WebCage(SuperGLEBerTaskModule):
    """WebCage Word Sense Disambiguation task.

    Task type: classification
    Evaluation metric: Accuracy
    """

    task_name = "webcage"
    task_group = "wsd"
    task_type = "classification"
    text_column = ["sentence_1", "sentence_2"]  # noqa: RUF012
    label_column = "labels"
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=2)
    num_classes = 2


TASK_MODULES = [
    OffensiveLanguage,
    ToxicComments,
    Polarity,
    DBAspect,
    HotelAspect,
    QueryAd,
    QuestAns,
    ParaphraseMatching,
    WebCage,
    VerbalIdioms,
    FactclaimingComments,
    EngagingComments,
    ArgumentMining,
    TopicRelevance,
    MassiveIntents,
    NLI,
    NewsClass,
    NERBiofid,
    NEREuroparl,
    NERWikiNews,
    NERLegal,
    NERNews,
    UPDep,
    UPPos,
    MassiveSeq,
    GermevalOpinions,
    PAWSX,
    MLQA,
    GermanQuad,
]
