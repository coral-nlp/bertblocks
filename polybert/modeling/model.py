from typing import TYPE_CHECKING, ClassVar

from polybert.modeling.packing import unpack_seq, pack_seq

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

from polybert.modeling.block import PolyBertBlock
from polybert.modeling.config import ConfigMixin, PolyBertConfig
from polybert.modeling.embedding import PolyBertEmbeddings
from polybert.modeling.head import PolyBertPredictionHead
from polybert.modeling.initialization import init_weights


class PolyBertPreTrainedModel(PreTrainedModel):
    config_class = PolyBertConfig
    base_model_prefix = "polybert"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_flex_attn = True

    def _init_weights(self, module: "nn.Module") -> None:
        self.apply(init_weights)


class PolyBertModel(PolyBertPreTrainedModel):
    def __init__(self, config: "PolyBertConfig"):
        super().__init__(config)
        self.embd = PolyBertEmbeddings(config)
        self.encd = PolyBertEncoder(config)
        self.num_heads = config.num_attention_heads
        self.post_init()

    def get_input_embeddings(self) -> "nn.Embedding":
        return self.embd.embd

    def set_input_embeddings(self, value: "nn.Embedding"):
        self.embd.embd = value

    def forward(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor | None" = None,
        output_attentions: "bool | None" = None,
        output_hidden_states: "bool | None" = False,
    ) -> "BaseModelOutput | BaseModelOutputWithPooling":
        x = self.embd(input_ids)
        x, hidden_states, attentions = self.encd(x, attention_mask, output_attentions, output_hidden_states)
        return BaseModelOutput(
            last_hidden_state=x,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=attentions if output_attentions else None,
        )


class PolyBertForMaskedLM(PolyBertPreTrainedModel):
    _tied_weight_keys: ClassVar = ["decoder.weight"]

    def __init__(self, config: "PolyBertConfig"):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.model = PolyBertModel(config)
        self.head = PolyBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        self.loss_fn = nn.CrossEntropyLoss()

        self.post_init()

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, new_embeddings: "nn.Linear"):
        self.decoder = new_embeddings

    def forward(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor | None" = None,
        position_ids: "torch.Tensor | None" = None,
        cu_seqlens: "torch.Tensor | None" = None,
        labels: "torch.Tensor | None" = None,
        output_attentions: "bool | None" = False,
        output_hidden_states: "bool | None" = False,
    ) -> "MaskedLMOutput":
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
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


class PolyBertForSequenceClassification(PolyBertPreTrainedModel):
    def __init__(self, config: "PolyBertConfig"):
        super().__init__(config)
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

    def forward(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor | None" = None,
        position_ids: "torch.Tensor | None" = None,
        cu_seqlens: "torch.Tensor | None" = None,
        labels: "torch.Tensor | None" = None,
        output_attentions: "bool | None" = False,
        output_hidden_states: "bool | None" = False,
    ) -> "SequenceClassifierOutput":
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        # Handle sequence packing for classification - extract first token of each sequence
        if self.config.enable_sequence_packing and cu_seqlens is not None:
            # Extract the first token (CLS) from each packed sequence
            batch_size = input_ids.shape[0]
            cls_tokens = []
            for i in range(batch_size):
                if i < len(cu_seqlens) - 1:
                    start_idx = cu_seqlens[i]
                    cls_tokens.append(output.last_hidden_state[i, start_idx])
                else:
                    cls_tokens.append(output.last_hidden_state[i, 0])
            cls_features = torch.stack(cls_tokens, dim=0)
        else:
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


class PolyBertForTokenClassification(PolyBertPreTrainedModel):
    def __init__(self, config: "PolyBertConfig"):
        super().__init__(config)
        self.model = PolyBertModel(config)
        self.head = PolyBertPredictionHead(config)
        self.num_classes = config.num_classes
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def forward(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor | None" = None,
        labels: "torch.Tensor | None" = None,
        output_attentions: "bool | None" = False,
        output_hidden_states: "bool | None" = False,
    ) -> "TokenClassifierOutput":
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


class PolyBertForQuestionAnswering(PolyBertPreTrainedModel):
    def __init__(self, config: "PolyBertConfig"):
        super().__init__(config)
        self.model = PolyBertModel(config)
        self.head = PolyBertPredictionHead(config)
        self.classifier = torch.nn.Linear(config.hidden_size, 2)
        self.post_init()

    def forward(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor | None" = None,
        start_positions: "torch.Tensor | None" = None,
        end_positions: "torch.Tensor | None" = None,
        output_attentions: "bool | None" = False,
        output_hidden_states: "bool | None" = False,
    ) -> "QuestionAnsweringModelOutput":
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
