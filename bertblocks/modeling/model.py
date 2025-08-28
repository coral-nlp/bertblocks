import functools
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from bertblocks.modeling.norms import get_norm

if TYPE_CHECKING:
    pass

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

from bertblocks.config import BertBlocksConfig
from bertblocks.modeling.backends import ATTENTION_BACKENDS
from bertblocks.modeling.block import Encoder
from bertblocks.modeling.embedding import TokenEmbedding
from bertblocks.modeling.head import Pooler, get_prediction_head
from bertblocks.modeling.loss import get_loss_function
from bertblocks.modeling.padding import pad_output, unpad_input


class BertBlocksPreTrainedModel(PreTrainedModel):
    """Base class for all BertBlocks models.

    This class provides the base configuration and weight initialization
    for all BertBlocks model variants. It inherits from HuggingFace's
    PreTrainedModel to provide compatibility with the transformers library.
    """

    config_class = BertBlocksConfig
    base_model_prefix = "bertblocks"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_flex_attn = True
    _no_split_modules: ClassVar[list] = ["Encoder", "Attention"]
    _keys_to_ignore_on_load_missing: ClassVar[list] = [r"position_ids"]
    _keys_to_ignore_on_load_unexpected: ClassVar[list] = [r"pooler"]

    def __init__(self, config: "BertBlocksConfig", *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)

    def _init_weights(self, module: "nn.Module") -> None:
        """Initialize module weights.

        Args:
            module: The module to initialize.

        """
        # Set up initialization parameters from config
        initializer_kind = self.config.initializer_kind
        initializer_cutoff_factor = self.config.initializer_cutoff_factor
        initializer_range = self.config.initializer_range
        initializer_gain = self.config.initializer_gain

        std_values = {
            "in": initializer_range,
            "out": initializer_range / math.sqrt(2.0 * self.config.num_blocks),
            "embedding": initializer_range,
            "final_out": self.config.hidden_size**-0.5,
        }

        # Determine std_kind based on module type
        std_kind = None

        # Embedding layers use "embedding" std
        if isinstance(module, nn.Embedding):
            std_kind = "embedding"
        # Linear layers - determine type by attribute name patterns
        elif isinstance(module, nn.Linear):
            # Get the full module path to determine context
            module_name = getattr(module, "_get_name", lambda: str(module))()

            # Task-specific output layers (final classification/generation layers)
            if any(name in module_name.lower() for name in ["classifier", "decoder"]):
                std_kind = "final_out"
            # Attention projection layers (input projections)
            elif any(name in module_name.lower() for name in ["proj", "uprj"]):
                std_kind = "in"
            # Feed-forward and output projections
            elif any(name in module_name.lower() for name in ["ffwd", "dprj"]):
                std_kind = "out"
            else:
                # Default for other linear layers
                std_kind = "in"

        if std_kind is None:
            return  # Skip initialization for unsupported modules

        std = std_values[std_kind]

        def _get_init_fn() -> "Callable[[torch.Tensor], None]":
            match initializer_kind:
                case "trunc_normal":
                    return functools.partial(
                        nn.init.trunc_normal_,
                        mean=0.0,
                        std=std,
                        a=-initializer_cutoff_factor * std,
                        b=initializer_cutoff_factor * std,
                    )
                case "kaiming_normal":
                    return functools.partial(nn.init.kaiming_normal_)
                case "kaiming_uniform":
                    return functools.partial(nn.init.kaiming_uniform_)
                case "xavier_normal":
                    return functools.partial(nn.init.xavier_normal_)
                case "xavier_uniform":
                    return functools.partial(nn.init.xavier_uniform_)
                case _:
                    raise ValueError(
                        f"Unknown initialization function {initializer_kind}, supported functions: "
                        f"'trunc_normal', 'kaiming_normal', 'kaiming_uniform', 'xavier_normal', 'xavier_uniform'"
                    )

        # Apply initialization function to module weight
        init_fn = _get_init_fn()
        if hasattr(module, "weight") and module.weight is not None:
            init_fn(module.weight)
            module.weight *= initializer_gain

        # Initialize bias terms to zero for linear layers
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)


class BertBlocksModel(BertBlocksPreTrainedModel):
    """Core BertBlocks model for encoding sequences.

    This is the base BertBlocks model that outputs hidden states without any
    task-specific head. It can be used as a feature extractor for downstream tasks.

    Attributes:
        embd (TokenEmbedding): Embedding layer.
        encd (Encoder): Encoder stack.
        norm (nn.Module): Normalization layer. Falls back to nn.Identity if not configured.
        pool (Pooler | None): Pooler layer, optional.
        pad_token_id (int): Token ID to insert for padding.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters. Passed to
            other submodules.
        add_pooling_layer (bool): Whether to add a pooling layer after the encoder layers.

    """

    def __init__(self, config: "BertBlocksConfig", add_pooling_layer: bool = False) -> None:
        super().__init__(config)
        self.unpadding = config.unpadding
        self.embd = TokenEmbedding(config)
        self.encd = Encoder(config)
        self.norm = get_norm(config) if config.include_final_norm else nn.Identity()
        self.pool = Pooler(config) if add_pooling_layer else None
        self.pad_token_id = config.pad_token_id or 0
        self.post_init()

    @property
    def dtype(self) -> "torch.dtype":
        """Get the dtype of the model parameters."""
        return next(self.parameters()).dtype

    @property
    def device(self) -> "torch.device":
        """Get the device of the model parameters."""
        return next(self.parameters()).device

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
        token_type_ids: "torch.Tensor | None" = None,
        output_attentions: "bool" = False,
        output_hidden_states: "bool" = False,
    ) -> "BaseModelOutput | BaseModelOutputWithPooling":
        """Forward pass of the BertBlocks model.

        Args:
            input_ids (torch.Tensor, shape [batch_size, seq_len]): Tensor of token ids.
            attention_mask (torch.Tensor | None, shape [batch_size, seq_len], optional): Tensor indicating which
                tokens should be attended to. Defaults to None.
            token_type_ids (torch.Tensor | None, shape [batch_size, seq_len], optional): Tensor indicating type
                of tokens. Defaults to None.
            output_attentions: Whether to return attention weights from all layers. Defaults to None.
            output_hidden_states: Whether to return hidden states from all layers. Defaults to False.

        Returns:
            BaseModelOutput or BaseModelOutputWithPooling containing:

                - `last_hidden_state`: Hidden states from the last layer
                - `pooler_output`: Pooler output from the last layer (optional)
                - `hidden_states`: Hidden states from all layers (optional)
                - `attentions`: Attention weights from all layers (optional)

        """
        B, S = input_ids.shape

        # Check if we need unpadded sequences (for FA2) or can work with padded sequences (SDPA/Native)
        backend = ATTENTION_BACKENDS[self.config.attn_implementation]

        if self.unpadding and backend.supports_unpadded:
            with torch.no_grad():
                input_ids_unpadded, indices, cu_seqlens, max_seq_len = unpad_input(
                    input_ids, attention_mask, self.pad_token_id
                )

            x = self.embd(input_ids_unpadded, token_type_ids=token_type_ids, cu_seqlens=cu_seqlens)

            x, hidden_states, attentions = self.encd(
                x, attention_mask, cu_seqlens, max_seq_len, output_attentions, output_hidden_states
            )
            x = self.norm(x)

            x = pad_output(x, indices, B, S)
            if output_hidden_states:
                hidden_states = [pad_output(h, indices, B, S, self.pad_token_id) for h in hidden_states]

        elif not self.unpadding and backend.supports_unpadded:
            x = self.embd(input_ids, token_type_ids=token_type_ids)

            x, hidden_states, attentions = self.encd(
                x, attention_mask, None, None, output_attentions, output_hidden_states
            )
            x = self.norm(x)

        else:
            raise ValueError("Model is loaded for unpadding mode, but attention backend does not support it.")

        if self.pool is not None:
            pooler_output = self.pool(x)
            return BaseModelOutputWithPooling(
                last_hidden_state=x,
                pooler_output=pooler_output,
                hidden_states=hidden_states if output_hidden_states else None,
                attentions=attentions if output_attentions else None,
            )
        else:
            return BaseModelOutput(
                last_hidden_state=x,
                hidden_states=hidden_states if output_hidden_states else None,
                attentions=attentions if output_attentions else None,
            )


class BertBlocksForTasksBase(BertBlocksPreTrainedModel):
    """Base class for all BertBlocks task-specific models.

    This class provides common functionality for classification, regression,
    and other downstream tasks, eliminating code duplication across task models.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters. Passed to
            other submodules.

    """

    def __init__(self, config: "BertBlocksConfig", *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        self.model = BertBlocksModel(config)
        self.head = get_prediction_head(config)

    def compute_loss(
        self,
        logits: "torch.Tensor",
        labels: "torch.Tensor",
        problem_type: "Literal['regression', 'single_label_classification', 'multi_label_classification'] | None",
    ) -> "torch.Tensor | None":
        """Compute loss for the given logits, labels and problem type.

        Args:
            logits: Model predictions
            labels: Target labels
            problem_type: Type of problem for loss computation

        Returns:
            torch.Tensor | None: Computed loss tensor or None if labels are not provided.

        """
        if labels is None:
            return None

        if problem_type == "regression":
            if self.num_labels == 1:
                return self.loss_fn(logits.squeeze(), labels.squeeze())
            else:
                return self.loss_fn(logits, labels)
        elif problem_type == "single_label_classification":
            return self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        elif problem_type == "multi_label_classification":
            return self.loss_fn(logits, labels.float())
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")


class BertBlocksForMaskedLM(BertBlocksPreTrainedModel):
    """BertBlocks model for masked language modeling tasks.

    This model extends the base BertBlocks model with a prediction head
    and decoder for masked language modeling. It can be used for
    pre-training or fine-tuning on masked language modeling tasks.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters. May be passed to
            other submodules. Keys used at top level:

            - `vocab_size`: Size of the vocabulary for token embeddings
            - `hidden_size`: Dimensionality of hidden layers

    """

    _tied_weight_keys: ClassVar = ["decoder.weight"]

    def __init__(self, config: "BertBlocksConfig"):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.model = BertBlocksModel(config)
        self.head = get_prediction_head(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        self.loss_fn = nn.CrossEntropyLoss()

        self.post_init()

    def get_input_embeddings(self) -> "nn.Module":
        """Return the encoder input embeddings."""
        return self.model.embd.embd

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
        token_type_ids: "torch.Tensor | None" = None,
        labels: "torch.Tensor | None" = None,
        output_attentions: "bool | None" = False,
        output_hidden_states: "bool | None" = False,
    ) -> "MaskedLMOutput":
        """Forward pass for masked language modeling.

        Args:
            input_ids (torch.Tensor, shape [batch_size, seq_len]): Tensor of token ids.
            attention_mask (torch.Tensor, shape [batch_size, seq_len], optional): Tensor indicating which tokens should
                be attended to. Defaults to None.
            token_type_ids (torch.Tensor, shape [batch_size, seq_len], optional): Tensor indicating type of tokens.
                Defaults to None.
            labels (torch.Tensor, shape [batch_size, seq_len], optional): Tensor of target token ids for computing loss.
                Defaults to None.
            output_attentions (bool): Whether to return attention weights from all layers. Defaults to None.
            output_hidden_states (bool): Whether to return hidden states from all layers. Defaults to False.

        Returns:
            MaskedLMOutput

                - `loss`: Masked language modeling loss if labels provided
                - `logits`: Prediction scores over vocabulary
                - `hidden_states`: Hidden states from all layers if requested
                - `attentions`: Attention weights from all layers if requested

        """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
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


class BertBlocksForSequenceClassification(BertBlocksForTasksBase):
    """BertBlocks model for sequence classification tasks.

    This model extends the base BertBlocks model with a classification head
    for sequence-level prediction tasks. It supports regression,
    single-label classification, and multi-label classification.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters. May be passed to
            other submodules. Keys used at top level:

            - `hidden_size`: Dimensionality of hidden layers
            - `num_labels`: Number of output labels for classification tasks
            - `problem_type`: Problem type for automatic loss selection

    """

    def __init__(self, config: "BertBlocksConfig"):
        super().__init__(config=config)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels
        self.problem_type = config.problem_type
        self.loss_fn = get_loss_function(self.problem_type)
        self.post_init()

    def forward(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor | None" = None,
        token_type_ids: "torch.Tensor | None" = None,
        labels: "torch.Tensor | None" = None,
        output_attentions: "bool | None" = False,
        output_hidden_states: "bool | None" = False,
    ) -> "SequenceClassifierOutput":
        """Forward pass for sequence classification.

        Args:
            input_ids (torch.Tensor, shape [batch_size, seq_len]): Tensor of token ids.
            attention_mask (torch.Tensor, shape [batch_size, seq_len], optional): Tensor indicating which tokens should
                be attended to. Defaults to None.
            token_type_ids (torch.Tensor, shape [batch_size, seq_len], optional): Tensor indicating type of tokens.
                Defaults to None.
            labels (torch.Tensor, shape [batch_size,] or [batch_size, num_labels], optional) : Tensor of target labels
                for computing loss.Defaults to None.
            output_attentions (bool): Whether to return attention weights from all layers. Defaults to None.
            output_hidden_states (bool): Whether to return hidden states from all layers. Defaults to False.

        Returns:
            SequenceClassifierOutput

                - `loss`: Classification loss if labels provided
                - `logits`: Classification scores
                - `hidden_states`: Hidden states from all layers if requested
                - `attentions`: Attention weights from all layers if requested

        """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        cls_features = output.last_hidden_state[:, 0, :]  # Regular CLS token extraction
        logits = self.classifier(self.head(cls_features))

        loss = self.compute_loss(logits, labels, self.problem_type)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


class BertBlocksForTokenClassification(BertBlocksForTasksBase):
    """BertBlocks model for token classification tasks.

    This model extends the base BertBlocks model with a classification head
    for token-level prediction tasks such as named entity recognition,
    part-of-speech tagging, and other sequence labeling tasks.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters. May be passed to
            other submodules. Keys used at top level:

            - `hidden_size`: Dimensionality of hidden layers
            - `num_labels`: Number of output labels for classification tasks

    """

    def __init__(self, config: "BertBlocksConfig"):
        super().__init__(config=config)
        self.num_labels = config.num_labels
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)
        # Token classification is always single-label classification; explicit literal cast is needed for mypy
        self.problem_type: Literal["single_label_classification"] = "single_label_classification"
        self.loss_fn = get_loss_function(self.problem_type)
        self.post_init()

    def forward(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor | None" = None,
        token_type_ids: "torch.Tensor | None" = None,
        labels: "torch.Tensor | None" = None,
        output_attentions: "bool | None" = False,
        output_hidden_states: "bool | None" = False,
    ) -> "TokenClassifierOutput":
        """Forward pass for token classification.

        Args:
            input_ids (torch.Tensor, shape [batch_size, seq_len]): Tensor of token ids.
            attention_mask (torch.Tensor, shape [batch_size, seq_len], optional): Tensor indicating which tokens should
                be attended to. Defaults to None.
            token_type_ids (torch.Tensor, shape [batch_size, seq_len], optional): Tensor indicating type of tokens.
                Defaults to None.
            labels (torch.Tensor, shape [batch_size, seq_len], optional) : Tensor of target labels for computing loss.
                Defaults to None.
            output_attentions (bool): Whether to return attention weights from all layers. Defaults to None.
            output_hidden_states (bool): Whether to return hidden states from all layers. Defaults to False.

        Returns:
            TokenClassifierOutput

                - `loss`: Token classification loss if labels provided
                - `logits`: Classification scores for each token
                - `hidden_states`: Hidden states from all layers if requested
                - `attentions`: Attention weights from all layers if requested

        """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits = self.classifier(self.head(output.last_hidden_state))

        loss = self.compute_loss(logits, labels, self.problem_type)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


class BertBlocksForQuestionAnswering(BertBlocksForTasksBase):
    """BertBlocks model for extractive question answering tasks.

    This model extends the base BertBlocks model with a classification head
    that predicts start and end positions of answers in the input sequence.
    It is designed for tasks like SQuAD where the answer is a span of text
    within the provided context.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters. May be passed to
            other submodules. Keys used at top level:

            - `hidden_size`: Dimensionality of hidden layers

    """

    def __init__(self, config: "BertBlocksConfig"):
        super().__init__(config=config)
        self.classifier = torch.nn.Linear(config.hidden_size, 2)  # start and end positions
        self.post_init()

    def forward(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor | None" = None,
        token_type_ids: "torch.Tensor | None" = None,
        start_positions: "torch.Tensor | None" = None,
        end_positions: "torch.Tensor | None" = None,
        output_attentions: "bool | None" = False,
        output_hidden_states: "bool | None" = False,
    ) -> "QuestionAnsweringModelOutput":
        """Forward pass for question answering.

        Args:
            input_ids (torch.Tensor, shape [batch_size, seq_len]): Tensor of token ids.
            attention_mask (torch.Tensor, shape [batch_size, seq_len], optional): Tensor indicating which tokens should
                be attended to. Defaults to None.
            token_type_ids (torch.Tensor, shape [batch_size, seq_len], optional): Tensor indicating type of tokens.
                Defaults to None.
            start_positions (torch.Tensor, shape [batch_size,], optional): Tensor of start positions for computing loss.
                Values should be in [0, sequence_length-1]. Defaults to None.
            end_positions (torch.Tensor, shape [batch_size,], optional): Tensor of end positions for computing loss.
                Values should be in [0, sequence_length-1]. Defaults to None.
            output_attentions (bool): Whether to return attention weights from all layers. Defaults to None.
            output_hidden_states (bool): Whether to return hidden states from all layers. Defaults to False.

        Returns:
            QuestionAnsweringModelOutput

                - `loss`: Span prediction loss if start_positions and end_positions provided
                - `start_logits`: Scores for start position of answer span
                - `end_logits`: Scores for end position of answer span
                - `hidden_states`: Hidden states from all layers if requested
                - `attentions`: Attention weights from all layers if requested

        """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
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
            # Sometimes the start/end positions are outside our model inputs, we ignore these terms
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


__all__ = [
    "BertBlocksModel",
    "BertBlocksForMaskedLM",
    "BertBlocksForTokenClassification",
    "BertBlocksForQuestionAnswering",
    "BertBlocksForSequenceClassification",
]
