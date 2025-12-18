import functools
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from einops import rearrange, repeat

from bertblocks.modeling.norms import get_norm
from bertblocks.modeling.scale import LayerScaler
from bertblocks.modeling.utils import LogLinearNoise

if TYPE_CHECKING:
    pass

import torch
import torch.nn.functional as F
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
from bertblocks.modeling.block import Encoder, EnhancedMaskingBlock, convert_to_4d_attention_mask
from bertblocks.modeling.embedding import TokenEmbedding
from bertblocks.modeling.head import Pooler, get_prediction_head
from bertblocks.modeling.loss import get_loss_function
from bertblocks.modeling.padding import pad_output, unpad_input
from bertblocks.modeling.position import AlibiPositionalEncoding


class BertBlocksPreTrainedModel(PreTrainedModel):
    """Base class for all BertBlocks models.

    This class provides the base configuration and weight initialization
    for all BertBlocks model variants. It inherits from HuggingFace's
    PreTrainedModel to provide compatibility with the transformers library.
    """

    config_class = BertBlocksConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
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
        self.unpadding = config._unpadding
        self.embd = TokenEmbedding(config)
        self.encd = Encoder(config)
        self.norm = get_norm(config) if config.include_final_norm else nn.Identity()
        self.scaler = LayerScaler(config.num_blocks) if config.norm_scaling else nn.Identity()
        self.pool = Pooler(config) if add_pooling_layer else None
        self.pad_token_id = config.pad_token_id or 0
        self.post_init()
        self.local_attention = config.local_attention
        self.alibi = (
            AlibiPositionalEncoding(config.num_attention_heads, device="cpu")
            if config.pos_emb_kind == "alibi"
            else None
        )

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
            attention_mask (torch.Tensor, shape [batch_size, seq_len], optional): Tensor indicating which
                tokens should be attended to. Defaults to None.
            token_type_ids (torch.Tensor, shape [batch_size, seq_len], optional): Tensor indicating type
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

        if self.unpadding:
            with torch.no_grad():
                input_ids, indices, cu_seqlens, max_seq_len = unpad_input(input_ids, attention_mask, self.pad_token_id)
            attention_mask = None
        else:
            indices, cu_seqlens, max_seq_len = None, None, None
            attention_mask = (
                torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
            )
            attention_mask = convert_to_4d_attention_mask(attention_mask)

            if self.config.pos_emb_kind == "alibi" and self.alibi is not None:
                attention_mask = self.alibi(attention_mask)

            if self.local_attention != (-1, -1) and self.local_attention[0] > 0:
                window_size = self.local_attention[0]
                pos = torch.arange(input_ids.shape[1], device=input_ids.device)
                local_mask = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs() <= window_size
                if attention_mask.dtype == torch.bool:
                    attention_mask = attention_mask & local_mask
                else:
                    attention_mask = attention_mask.masked_fill(~local_mask, -float("inf"))

        # Input embeddings
        x = self.embd(input_ids, token_type_ids=token_type_ids, cu_seqlens=cu_seqlens)

        x, hidden_states, attentions = self.encd(
            x, attention_mask, cu_seqlens, max_seq_len, output_attentions, output_hidden_states
        )
        x = self.norm(x)
        x = self.scaler(x)

        if self.config._unpadding:
            x = pad_output(x, indices, B, S)
            if output_hidden_states:
                hidden_states = [pad_output(h, indices, B, S, self.pad_token_id) for h in hidden_states]

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
            if self.num_classes == 1:
                return self.loss_fn(logits.squeeze(), labels.squeeze())
            else:
                return self.loss_fn(logits, labels)
        elif problem_type == "single_label_classification":
            return self.loss_fn(logits.view(-1, self.num_classes), labels.view(-1))
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

    _tied_weights_keys: ClassVar = ["decoder.weight"]

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


class BertBlocksForEnhancedMaskedLM(BertBlocksForMaskedLM):
    """BertBlocks model for enhanced masked language modeling tasks.

    This model extends the base BertBlocks model with a prediction head
    and decoder for enhanced masked language modeling. It can be used for
    pre-training or fine-tuning on enhanced masked language modeling tasks.
    Enhanced masked language modeling uses one additional transformer layer
    to handle the masking, instead of masking input tokens.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters. May be passed to
            other submodules. Keys used at top level:

            - `vocab_size`: Size of the vocabulary for token embeddings
            - `hidden_size`: Dimensionality of hidden layers
        masking_strategy (str): Masking strategy to use.
            Available options: "random".
        masking_probability (float): Probability of masking tokens. Defaults to 0.5.
    """

    _tied_weights_keys: ClassVar = ["decoder.weight"]

    def __init__(
        self,
        config: "BertBlocksConfig",
        masking_strategy: Literal["random"] = "random",
        masking_probability: float = 0.5,
    ):
        super().__init__(config)
        self.enhanced_masking_block = EnhancedMaskingBlock(
            config, config.num_blocks + 1, masking_strategy, masking_probability
        )
        self.post_init()

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
        hidden_state, _ = self.enhanced_masking_block(output.last_hidden_state, attention_mask)
        logits = self.decoder(self.head(hidden_state))

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
            - `num_classes`: Number of output labels for classification tasks
            - `problem_type`: Problem type for automatic loss selection

    """

    def __init__(self, config: "BertBlocksConfig"):
        super().__init__(config=config)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_classes)
        self.num_classes = config.num_classes
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
            labels (torch.Tensor, shape [batch_size,] or [batch_size, num_classes], optional) : Tensor of target labels
                for computing loss. Defaults to None.
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
            - `num_classes`: Number of output labels for classification tasks

    """

    def __init__(self, config: "BertBlocksConfig"):
        super().__init__(config=config)
        self.num_classes = config.num_classes
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_classes)
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


class BertBlocksForMaskedDiffusion(BertBlocksForMaskedLM):
    """Implementation of a masked diffusion model.

    Closely follows https://github.com/kuleshov-group/mdlm
    """

    def __init__(self, config: "BertBlocksConfig"):
        super().__init__(config)
        self.max_seq_len = config.max_sequence_length
        self.mask_token_id = config.mask_token_id
        # Noise Schedule
        self.noise = LogLinearNoise()
        self.post_init()

    def get_input_embeddings(self) -> "nn.Module":
        """Return the encoder input embeddings."""
        return self.model.model.embd.embd

    def get_output_embeddings(self) -> "nn.Module":
        """Return the decoder embeddings."""
        return self.model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: "nn.Linear") -> None:
        """Replace the decoder embeddings with given one (e.g., the encoder side)."""
        self.model.decoder = new_embeddings

    def forward(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor | None" = None,
        token_type_ids: "torch.Tensor | None" = None,
        labels: "torch.Tensor | None" = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
    ) -> "MaskedLMOutput":
        """Forward pass for diffusion language modeling.

        Args:
            input_ids (torch.Tensor, shape [batch_size, seq_len]): Tensor of token ids. When training, should
                be timestep-corrupted token IDs.
            attention_mask (torch.Tensor, shape [batch_size, seq_len], optional): Tensor indicating which tokens should
                be attended to. Defaults to None (all tokens are attended to).
            token_type_ids (torch.Tensor, shape [batch_size, seq_len], optional): Tensor indicating type of tokens.
                Defaults to None.
            labels (torch.Tensor, shape [batch_size, seq_len], optional): Tensor indicating uncorrupted token IDs.
            output_attentions (bool): Whether to return attention weights from all layers. Defaults to False.
            output_hidden_states (bool): Whether to return hidden states from all layers. Defaults to False.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        output: MaskedLMOutput = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if labels is not None:
            logits = self._process_logits(input_ids=input_ids, logits=output.logits)
            # Get the per-token log-likelihood of ground-truth tokens
            token_nll = torch.gather(input=logits, dim=-1, index=labels[:, :, None]).squeeze(-1)
            # Negate loss to optimize in correct direction
            loss = -1 * token_nll
            # Average weighted NLL over valid token positions
            loss = (loss * attention_mask).sum() / attention_mask.sum()
        else:
            logits = output.logits
            loss = None

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    def _prior(self, *batch_dims: Sequence[int]) -> torch.Tensor:
        """Calculate the prior distribution, i.e., a fully masked tensor.

        Args:
            batch_dims (Sequence[int]): Sequence of batch dimensions.

        Returns:
            torch.Tensor, shape [*batch_dims]: Fully masked tensor.
        """
        return self.mask_token_id * torch.ones(*batch_dims, dtype=torch.int64)

    def _sample_categorical(self, categorical_probs: torch.Tensor) -> torch.Tensor:
        """Sample from categorical distribution using Gumbel-max trick.

        Works with unnormalized probabilities/weights. The Gumbel-max trick samples
        by adding Gumbel noise and taking the argmax, which is equivalent to sampling
        from the categorical distribution but more numerically stable and efficient.

        Args:
            categorical_probs (torch.Tensor, shape [batch, seq_len, vocab_size]):
                Unnormalized probabilities or weights.

        Returns:
            torch.Tensor (shape [batch, seq_len]): Sampled token indices.
        """
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)

    @torch.no_grad()
    def sample(
        self,
        input_ids: "torch.Tensor | None" = None,
        attention_mask: "torch.Tensor | None" = None,
        max_length: int | None = None,
        num_samples: int = 1,
        num_steps: int = 100,
        eps: float = 1e-5,
    ) -> "torch.Tensor":
        """Generate samples using iterative denoising from noise to data.

        Supports both unconditional generation and prefix-conditioned generation.
        Compatible with HuggingFace tokenizer output.

        Args:
            input_ids (torch.Tensor, shape [batch_size, seq_len], optional): Input token IDs to condition on.
                If provided, tokens where attention_mask=1 will be preserved during sampling.
                If None, generates unconditionally from scratch.
            attention_mask (torch.Tensor, shape [batch_size, seq_len], optional): Mask indicating which
                input_ids positions to preserve (1) vs denoise (0). If None and input_ids is provided,
                all input positions are preserved.
            max_length (int, optional): Maximum sequence length to generate. If None, uses self.max_seq_len.
                If input_ids is shorter than max_length, extends with MASK tokens.
            num_samples (int, optional): Number of sequences to generate when input_ids is None. Defaults to 1.
            num_steps (int, optional): Number of denoising steps (more = higher quality, slower). Defaults to 100.
            eps (float, optional): Final noise level. Defaults to 1e-5.

        Returns:
            torch.Tensor (shape [batch_size, max_length] or [num_samples, max_length]): Generated token sequences.

        Examples:
            # Unconditional generation
            >>> sequences = model.sample(num_samples=4, max_length=128)

            # Prefix-conditioned generation
            >>> inputs = tokenizer("The cat sat on", return_tensors="pt")
            >>> sequences = model.sample(**inputs, max_length=128)
        """
        # Determine target length
        target_length = max_length or self.max_seq_len

        # Initialize sequence
        if input_ids is None:
            # Unconditional generation: start from all MASK tokens
            x = self._prior((num_samples, target_length)).to(self.device)
            prefix_mask = None
        else:
            # Conditional generation: use provided prefix
            batch_size, seq_len = input_ids.shape
            input_ids = input_ids.to(self.device)

            # Create prefix mask (which positions to preserve)
            if attention_mask is None:
                # If no mask provided, preserve all input tokens
                prefix_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
            else:
                prefix_mask = attention_mask.bool().to(self.device)
                # Ensure attention_mask matches input_ids length
                assert (
                    prefix_mask.shape[1] == seq_len
                ), f"attention_mask length {prefix_mask.shape[1]} doesn't match input_ids length {seq_len}"

            if seq_len < target_length:
                # Extend to target length if needed
                # Pad input_ids with MASK tokens
                padding = self._prior((batch_size, target_length - seq_len)).to(self.device)
                x = torch.cat([input_ids, padding], dim=1)
                # Extend prefix_mask with False (denoise padded positions)
                mask_padding = torch.zeros(batch_size, target_length - seq_len, dtype=torch.bool, device=self.device)
                prefix_mask = torch.cat([prefix_mask, mask_padding], dim=1)
            elif seq_len > target_length:
                # Truncate if input is longer than target
                x = input_ids[:, :target_length]
                prefix_mask = prefix_mask[:, :target_length]
            else:
                x = input_ids.clone()
                # prefix_mask already has correct shape (seq_len == target_length)

            # Ensure prefix_mask matches x's shape (defensive check)
            assert prefix_mask.shape == x.shape, f"Shape mismatch: prefix_mask {prefix_mask.shape} vs x {x.shape}"

            # Set non-prefix positions to MASK
            x[~prefix_mask] = self.mask_token_id

        # Create schedule: [1.0, ..., eps]
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        step_size = (1 - eps) / num_steps

        # Store original prefix for restoration
        if prefix_mask is not None:
            original_prefix = x.clone()

        # Iterative denoising loop
        for t in timesteps[:-1]:  # All except last
            time = repeat(t.unsqueeze(-1), "1 -> b 1", b=x.shape[0])
            x = self._reverse_step(x, time, step_size)

            # Restore prefix tokens after each step
            if prefix_mask is not None:
                x[prefix_mask] = original_prefix[prefix_mask]

        # Final cleanup (greedy sample any remaining mask tokens)
        x = self._final_denoise(x)

        # Restore prefix one last time
        if prefix_mask is not None:
            x[prefix_mask] = original_prefix[prefix_mask]

        return x

    def _process_logits(self, input_ids: "torch.Tensor", logits: "torch.Tensor") -> "torch.Tensor":
        # Set the prediction probability of the mask token to 0 (-inf in log space)
        logits[:, :, self.mask_token_id] = -torch.inf
        # Re-normalize the logits such that x.exp() is a probability distribution over vocab_size.
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        # Clamp unmasked ground-truth tokens (p=1 for ground-truth token, p=0 for any other token at that position)
        unmasked_indices = input_ids != self.mask_token_id
        logits[unmasked_indices] = -torch.inf
        logits[unmasked_indices, input_ids[unmasked_indices]] = 0
        return logits

    def _compute_transition_probs(self, input_ids: "torch.Tensor", sigma: "torch.Tensor") -> "torch.Tensor":
        """Compute token-wise transition probabilities for discrete diffusion reverse process.

        Score represents transition probability ratios:
            P(x_t, y) = P(y -> x_t) / P(x_t -> x_t)

        Two cases:
        1. x_t = MASK: Use model predictions scaled by noise schedule
        2. x_t = token: Only allow staying same or coming from MASK

        Args:
            input_ids (torch.Tensor, shape [batch_size, seq_len]): Input sequences.
            sigma (torch.Tensor, shape [batch_size, 1]): Tensor indicating timestep sigma of each sequence.

        Returns:
            torch.Tensor (shape [batch_size, seq_len, vocab_size]): Tensor of transition probabilities for each token.
        """
        # == q_t
        # Pass timestep here, as it will internally convert to sigma
        logits = self._forward(input_ids=input_ids).logits
        logits = self._process_logits(input_ids=input_ids, logits=logits)
        batch, seq, vocab = logits.shape

        # k = P(token = [MASK]) / P(token = ~[MASK]) = exp(- timestep) / (1 - exp(- timestep))
        # Probability ratio between masked and unmasked states == 1/k in log space; used as logit scaling factor
        log_k = -torch.log(torch.expm1(sigma))  # [batch, 1, 1], negative value
        log_k = rearrange(log_k, "b 1 1 -> b")

        # score(x, t) = p_t(y) / p_t(x) = log p_t(y) - log p_t(x) = logits + log_k (log_k is already negative)
        # Case 1: Original token was [MASK]
        # Case 1.1: Predicted token is not [MASK] -> score = prob ratio
        masked_log_score = logits + repeat(log_k, "b -> b 1 1")
        # Case 1.2: Predicted token is [MASK] -> score = equilibrium = 0 (stay here)
        masked_log_score[:, :, self.mask_token_id] = 0

        # Case 2: Original token was not mask [MASK]
        # Case 2.1: Predicted token is not original token and not mask token -> score = ... / 0 (never move here)
        unmasked_log_score = torch.full_like(logits, -torch.inf)
        # Case 2.2: Predicted token is original token -> score = equilibrium = 0 (stay here)
        unmasked_log_score.scatter_(dim=-1, index=repeat(input_ids, "b s -> b s 1"), value=0.0)
        # Can 2.3: Predicted token is not original token, but is mask token -> score = 1/k (maybe move here)
        unmasked_log_score[:, :, self.mask_token_id] = -repeat(log_k, "b -> b s", s=seq)

        # Combine based on whether position is masked
        is_masked = repeat((input_ids == self.mask_token_id), "b s -> b s v", v=vocab)
        log_score = torch.where(is_masked, masked_log_score, unmasked_log_score)
        # Transfer from log space and return
        return log_score.exp()

    def _staggered_correction(self, transition_probs: "torch.Tensor", dsigma: "torch.Tensor") -> "torch.Tensor":
        """Apply staggered probability correction for absorbing state diffusion.

        Args:
            transition_probs (torch.Tensor, shape [batch_size, seq_len, vocab]): Transition probabilities.
            dsigma (torch.Tensor, shape [batch_size, 1, 1]): Noise level change (positive when denoising).

        Returns:
            torch.Tensor (shape [batch_size, seq_len, vocab]): Corrected transition probabilities.
        """
        transition_probs = transition_probs.clone()
        dsigma = torch.exp(dsigma)

        # Compute extra mass to redirect to MASK token (based on original scores before scaling)
        extra_const = (1 - dsigma.squeeze(-1)) * transition_probs.sum(dim=-1)

        # Scale all scores by exp(dsigma)
        transition_probs = transition_probs * dsigma

        # Add the redirected mass to MASK token
        transition_probs[..., self.mask_token_id] += extra_const

        return transition_probs

    def _reverse_step(self, input_ids: "torch.Tensor", timestep: "torch.Tensor", step_size: float) -> "torch.Tensor":
        """Perform a single reverse diffusion step through ancestral sampling.

        Args:
            input_ids (torch.Tensor, shape [batch_size, seq_len]): Input sequences.
            timestep (torch.Tensor, shape [batch_size, 1]): Time step for each sequence (t in [0,1]).
            step_size (float): Step size to move in time.

        Returns:
            torch.Tensor (shape [batch_size, seq_len]): Partially step-wise denoised sequences.
        """
        # Get alpha values at current and next timesteps
        _, curr_alpha = self.noise(timestep)
        _, next_alpha = self.noise(timestep - step_size)

        # Convert to sigma for model conditioning and kernels
        curr_sigma = -torch.log(curr_alpha).unsqueeze(-1)
        next_sigma = -torch.log(next_alpha).unsqueeze(-1)
        dsigma = curr_sigma - next_sigma

        # Get transition_probs and apply corrections
        transition_probs = self._compute_transition_probs(input_ids, curr_sigma)
        transition_probs = self._staggered_correction(transition_probs, dsigma)
        # Compute posterior
        posterior_distribution = transition_probs * self._transp_transition(input_ids, dsigma)
        # Sample using Gumbel-max trick (works with unnormalized probabilities)
        return self._sample_categorical(posterior_distribution)

    def _final_denoise(self, input_ids: "torch.Tensor") -> "torch.Tensor":
        """Perform final greedy denoising step at near-zero noise to clean output.

        Args:
            input_ids (torch.Tensor, shape [batch_size, seq_len]): Input sequences.
            timestep (torch.Tensor, shape [batch_size, 1]): Time step for each sequence (t in [0,1]).

        Returns:
            torch.Tensor (shape [batch_size, seq_len]): Final denoised sequences.
        """
        # Infer final logits
        logits = self._forward(input_ids).logits
        logits = self._process_logits(input_ids=input_ids, logits=logits)

        # For MASK positions, use model predictions, for non-MASK positions, keep current token
        is_mask = input_ids == self.mask_token_id
        posterior_distribution = logits.exp()  # Convert log probs to probs

        # For non-mask positions, create one-hot distribution at current token
        non_mask_dist = torch.zeros_like(posterior_distribution)
        non_mask_dist.scatter_(dim=-1, index=input_ids.unsqueeze(-1), value=1.0)

        # Combine: use model predictions for MASK, keep current for non-MASK
        posterior_distribution = torch.where(
            is_mask.unsqueeze(-1).expand_as(posterior_distribution), posterior_distribution, non_mask_dist
        )

        # Sample using Gumbel-max trick
        return self._sample_categorical(posterior_distribution)

    def _transp_transition(self, state: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the transpose transition kernel.

        Indicates the probability of tokens at a position at prior time given the observed token at current time.

        Args:
            state (torch.Tensor, shape [batch_size, seq_len]): observed tokens at current time.
            sigma (torch.Tensor, shape [batch_size, 1, 1]): Tensor indicating noise level of each sequence.

        Returns:
            torch.Tensor (shape [batch_size, seq_len, vocab_size]): transition probabilities for tokens into prior time.
        """
        stay_prob = torch.exp(-sigma).squeeze(-1)

        # When observing a token, assume it came from itself
        # P(was token i | observe token i) = exp(-sigma)
        transition_prob = stay_prob * F.one_hot(state, num_classes=self.vocab_size)

        # When observing MASK, add flow from all tokens that could have transitioned to MASK
        # P(was any token | observe MASK) gets additional (1 - exp(-sigma)) probability
        is_mask = state == self.mask_token_id
        flow_to_mask = torch.where(is_mask, 1 - stay_prob.squeeze(-1), torch.zeros_like(is_mask, dtype=stay_prob.dtype))

        return transition_prob + flow_to_mask.unsqueeze(-1)


def get_model_cls(
    task: Literal[
        "mlm", "diffusion", "enhanced_mlm", "denoising", "classification", "token_classification", "question_answering"
    ],
) -> type[
    BertBlocksForMaskedLM
    | BertBlocksForMaskedDiffusion
    | BertBlocksForEnhancedMaskedLM
    | BertBlocksForSequenceClassification
    | BertBlocksForTokenClassification
    | BertBlocksForQuestionAnswering
]:
    match task:
        case "mlm":
            return BertBlocksForMaskedLM
        case "diffusion":
            return BertBlocksForMaskedDiffusion
        case "enhanced_mlm":
            return BertBlocksForEnhancedMaskedLM
        case "sequence_classification":
            return BertBlocksForSequenceClassification
        case "token_classification":
            return BertBlocksForTokenClassification
        case "question_answering":
            return BertBlocksForQuestionAnswering
        case _:
            raise ValueError(
                f"Unknown task {task}, expected one of 'mlm', 'diffusion', 'enhanced_mlm', 'sequence_classification', "
                f"'token_classification', 'question_answering'"
            )


__all__ = [
    "BertBlocksForMaskedDiffusion",
    "BertBlocksForMaskedLM",
    "BertBlocksForQuestionAnswering",
    "BertBlocksForSequenceClassification",
    "BertBlocksForTokenClassification",
    "BertBlocksModel",
]
