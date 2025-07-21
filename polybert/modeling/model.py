from typing import TYPE_CHECKING, ClassVar

from polybert.modeling.block import PolyBertEncoder

if TYPE_CHECKING:
    import torch

from torch import nn
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel

from polybert.modeling.config import PolyBertConfig
from polybert.modeling.embedding import PolyBertEmbeddings
from polybert.modeling.head import PolyBertPredictionHead
from polybert.modeling.initialization import InitMixin


class PolyBertPreTrainedModel(PreTrainedModel):
    """Base class for all PolyBert models.

    This class provides the base configuration and weight initialization
    for all PolyBert model variants. It inherits from HuggingFace's
    PreTrainedModel to provide compatibility with the transformers library.

    Attributes:
        config_class: Configuration class for PolyBert models.
        base_model_prefix: Prefix used for model state dict keys.
        supports_gradient_checkpointing: Whether gradient checkpointing is supported.
        _supports_flash_attn_2: Whether Flash Attention 2 is supported.
        _supports_sdpa: Whether Scaled Dot Product Attention is supported.
        _supports_flex_attn: Whether Flexible Attention is supported.

    """

    config_class = PolyBertConfig
    base_model_prefix = "polybert"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_flex_attn = True


class PolyBertModel(PolyBertPreTrainedModel):
    """Core PolyBert model for encoding sequences.

    This is the base PolyBert model that outputs hidden states without any
    task-specific head. It can be used as a feature extractor for downstream tasks.

    The model consists of:
    - Embedding layer for token, position, and type embeddings
    - Stack of transformer encoder blocks
    - Optional pooling layer
    """

    def __init__(self, config: "PolyBertConfig"):
        """Initialize the PolyBert model.

        Args:
            config: Model configuration containing hyperparameters.

        """
        super().__init__(config)
        self.embd = PolyBertEmbeddings(config)
        self.encd = PolyBertEncoder(config)
        self.num_heads = config.num_attention_heads
        self.post_init()

    def get_input_embeddings(self) -> "nn.Embedding":
        """Get the input token embeddings.

        Returns:
            nn.Embedding: The input token embedding layer.

        """
        return self.embd.embd

    def set_input_embeddings(self, value: "nn.Embedding") -> None:
        """Set the input token embeddings.

        Args:
            value: The new input token embedding layer to use.

        """
        self.embd.embd = value

    def forward(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor | None" = None,
        output_attentions: "bool | None" = None,
        output_hidden_states: "bool | None" = False,
    ) -> "BaseModelOutput | BaseModelOutputWithPooling":
        """Forward pass through the PolyBert model.

        Args:
            input_ids: Tensor of token ids of shape (batch_size, sequence_length).
            attention_mask: Optional tensor indicating which tokens should be attended to.
                Shape (batch_size, sequence_length). Defaults to None.
            output_attentions: Whether to return attention weights from all layers.
                Defaults to None.
            output_hidden_states: Whether to return hidden states from all layers.
                Defaults to False.

        Returns:
            BaseModelOutput containing:
                - last_hidden_state: Hidden states from the last layer
                - hidden_states: Hidden states from all layers if requested
                - attentions: Attention weights from all layers if requested

        """
        x = self.embd(input_ids)
        x, hidden_states, attentions = self.encd(x, attention_mask, output_attentions, output_hidden_states)
        return BaseModelOutput(
            last_hidden_state=x,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=attentions if output_attentions else None,
        )


class PolyBertForMaskedLM(PolyBertPreTrainedModel, InitMixin):
    """PolyBert model for masked language modeling tasks.

    This model extends the base PolyBert model with a prediction head
    and decoder for masked language modeling. It can be used for
    pre-training or fine-tuning on masked language modeling tasks.

    Attributes:
        _tied_weight_keys: List of weight keys that should be tied between modules.

    """

    _tied_weight_keys: ClassVar = ["decoder.weight"]

    def __init__(self, config: "PolyBertConfig"):
        """Initialize the PolyBert masked language model.

        Args:
            config: Model configuration containing hyperparameters.

        """
        super(PolyBertPreTrainedModel, self).__init__(config)
        super(InitMixin, self).__init__(config)
        self.vocab_size = config.vocab_size
        self.model = PolyBertModel(config)
        self.head = PolyBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        self.loss_fn = nn.CrossEntropyLoss()

        self.post_init()

    def init_weights(self) -> None:
        """Initialize weights."""
        self._init_module_weights(self.decoder, "out")

    def get_output_embeddings(self) -> "nn.Module":
        """Return the decoder embeddings."""
        return self.decoder

    def set_output_embeddings(self, new_embeddings: "nn.Linear") -> None:
        """Replace the decoder embeddings with given one (e.g., the encoder side)."""
        self.decoder = new_embeddings

    def forward(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor | None" = None,
        labels: "torch.Tensor | None" = None,
        output_attentions: "bool | None" = False,
        output_hidden_states: "bool | None" = False,
    ) -> "MaskedLMOutput":
        """Forward pass for masked language modeling.

        Args:
            input_ids: Tensor of token ids of shape (batch_size, sequence_length).
            attention_mask: Optional tensor indicating which tokens should be attended to.
                Shape (batch_size, sequence_length). Defaults to None.
            labels: Optional tensor of target token ids for computing loss.
                Shape (batch_size, sequence_length). Defaults to None.
            output_attentions: Whether to return attention weights from all layers.
                Defaults to False.
            output_hidden_states: Whether to return hidden states from all layers.
                Defaults to False.

        Returns:
            MaskedLMOutput containing:
                - loss: Masked language modeling loss if labels provided
                - logits: Prediction scores over vocabulary
                - hidden_states: Hidden states from all layers if requested
                - attentions: Attention weights from all layers if requested

        """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits = self.decoder(self.head(output.last_hidden_state))

        loss = None
        if labels is not None:
            labels = labels.flatten()  # (b d -> (b d))
            logits = logits.flatten(0, 1)  # (b d v -> (b d) v)
            loss = self.loss_fn(logits, labels)

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


class PolyBertForSequenceClassification(PolyBertPreTrainedModel, InitMixin):
    """PolyBert model for sequence classification tasks.

    This model extends the base PolyBert model with a classification head
    for sequence-level prediction tasks. It supports regression,
    single-label classification, and multi-label classification.
    """

    def __init__(self, config: "PolyBertConfig"):
        """Initialize the PolyBert sequence classification model.

        Args:
            config: Model configuration containing hyperparameters including
                task type ('regression', 'single_label_classification', or
                'multi_label_classification') and number of classes.

        Raises:
            ValueError: If the task type is not supported.

        """
        super(PolyBertPreTrainedModel, self).__init__(config)
        super(InitMixin, self).__init__(config)
        self.model = PolyBertModel(config)
        self.head = PolyBertPredictionHead(config)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_classes)
        self.num_classes = config.num_classes
        if self.config.task == "regression":
            self.loss_fn = nn.MSELoss()
        elif self.config.task == "single_label_classification":
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.config.task == "multi_label_classification":
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(
                "Unknown problem type {self.config.problem_type}, "
                "supported are 'regression', 'single_label_classification', "
                "and 'multi_label_classification'."
            )
        self.post_init()

    def init_weights(self) -> None:
        """Initialize weights."""
        self._init_module_weights(self.classifier, "final_out")

    def forward(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor | None" = None,
        labels: "torch.Tensor | None" = None,
        output_attentions: "bool | None" = False,
        output_hidden_states: "bool | None" = False,
    ) -> "SequenceClassifierOutput":
        """Forward pass for sequence classification.

        Args:
            input_ids: Tensor of token ids of shape (batch_size, sequence_length).
            attention_mask: Optional tensor indicating which tokens should be attended to.
                Shape (batch_size, sequence_length). Defaults to None.
            labels: Optional tensor of target labels for computing loss.
                For regression: shape (batch_size,) or (batch_size, num_classes).
                For classification: shape (batch_size,) for single-label or
                (batch_size, num_classes) for multi-label. Defaults to None.
            output_attentions: Whether to return attention weights from all layers.
                Defaults to False.
            output_hidden_states: Whether to return hidden states from all layers.
                Defaults to False.

        Returns:
            SequenceClassifierOutput containing:
                - loss: Classification loss if labels provided
                - logits: Classification scores
                - hidden_states: Hidden states from all layers if requested
                - attentions: Attention weights from all layers if requested

        """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        cls_features = output.last_hidden_state[:, 0, :]  # Regular CLS token extraction
        logits = self.classifier(self.head(cls_features))

        loss = None
        if labels is not None:
            if self.config.task == "regression":
                if self.num_classes == 1:
                    loss = self.loss_fn(logits.squeeze(), labels.squeeze())
                else:
                    loss = self.loss_fn(logits, labels)
            elif self.config.task == "single_label_classification":
                loss = self.loss_fn(logits.view(-1, self.num_classes), labels.view(-1))
            elif self.config.task == "multi_label_classification":
                loss = self.loss_fn(logits, labels.float())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


class PolyBertForTokenClassification(PolyBertPreTrainedModel, InitMixin):
    """PolyBert model for token classification tasks.

    This model extends the base PolyBert model with a classification head
    for token-level prediction tasks such as named entity recognition,
    part-of-speech tagging, and other sequence labeling tasks.
    """

    def __init__(self, config: "PolyBertConfig"):
        """Initialize the PolyBert token classification model.

        Args:
            config: Model configuration containing hyperparameters including
                the number of classes for token classification.

        """
        super(PolyBertPreTrainedModel, self).__init__(config)
        super(InitMixin, self).__init__(config)
        self.model = PolyBertModel(config)
        self.head = PolyBertPredictionHead(config)
        self.num_classes = config.num_classes
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def init_weights(self) -> None:
        """Initialize weights."""
        self._init_module_weights(self.classifier, "final_out")

    def forward(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor | None" = None,
        labels: "torch.Tensor | None" = None,
        output_attentions: "bool | None" = False,
        output_hidden_states: "bool | None" = False,
    ) -> "TokenClassifierOutput":
        """Forward pass for token classification.

        Args:
            input_ids: Tensor of token ids of shape (batch_size, sequence_length).
            attention_mask: Optional tensor indicating which tokens should be attended to.
                Shape (batch_size, sequence_length). Defaults to None.
            labels: Optional tensor of target token labels for computing loss.
                Shape (batch_size, sequence_length). Defaults to None.
            output_attentions: Whether to return attention weights from all layers.
                Defaults to False.
            output_hidden_states: Whether to return hidden states from all layers.
                Defaults to False.

        Returns:
            TokenClassifierOutput containing:
                - loss: Token classification loss if labels provided
                - logits: Classification scores for each token
                - hidden_states: Hidden states from all layers if requested
                - attentions: Attention weights from all layers if requested

        """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        logits = self.classifier(self.head(output.last_hidden_state))

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_classes), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


class PolyBertForQuestionAnswering(PolyBertPreTrainedModel, InitMixin):
    """PolyBert model for extractive question answering tasks.

    This model extends the base PolyBert model with a classification head
    that predicts start and end positions of answers in the input sequence.
    It is designed for tasks like SQuAD where the answer is a span of text
    within the provided context.
    """

    def __init__(self, config: "PolyBertConfig"):
        """Initialize the PolyBert question answering model.

        Args:
            config: Model configuration containing hyperparameters.

        """
        super(PolyBertPreTrainedModel, self).__init__(config)
        super(InitMixin, self).__init__(config)
        self.model = PolyBertModel(config)
        self.head = PolyBertPredictionHead(config)
        self.classifier = torch.nn.Linear(config.hidden_size, 2)
        self.post_init()

    def init_weights(self) -> None:
        """Initialize weights."""
        self._init_module_weights(self.classifier, "final_out")

    def forward(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor | None" = None,
        start_positions: "torch.Tensor | None" = None,
        end_positions: "torch.Tensor | None" = None,
        output_attentions: "bool | None" = False,
        output_hidden_states: "bool | None" = False,
    ) -> "QuestionAnsweringModelOutput":
        """Forward pass for question answering.

        Args:
            input_ids: Tensor of token ids of shape (batch_size, sequence_length).
            attention_mask: Optional tensor indicating which tokens should be attended to.
                Shape (batch_size, sequence_length). Defaults to None.
            start_positions: Optional tensor of start positions for computing loss.
                Shape (batch_size,). Values should be in [0, sequence_length-1].
                Defaults to None.
            end_positions: Optional tensor of end positions for computing loss.
                Shape (batch_size,). Values should be in [0, sequence_length-1].
                Defaults to None.
            output_attentions: Whether to return attention weights from all layers.
                Defaults to False.
            output_hidden_states: Whether to return hidden states from all layers.
                Defaults to False.

        Returns:
            QuestionAnsweringModelOutput containing:
                - loss: Span prediction loss if start_positions and end_positions provided
                - start_logits: Scores for start position of answer span
                - end_logits: Scores for end position of answer span
                - hidden_states: Hidden states from all layers if requested
                - attentions: Attention weights from all layers if requested

        Note:
            The loss is computed as the average of start and end position losses.
            Positions outside the input sequence are clamped and ignored during loss computation.

        """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        logits = self.classifier(self.head(output.last_hidden_state))

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            start_loss = nn.functional.cross_entropy(start_logits, start_positions, ignore_index=ignored_index)
            end_loss = nn.functional.cross_entropy(end_logits, end_positions, ignore_index=ignored_index)
            loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )
