from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import Seq2SeqLMOutput

# Use in model/reflect.py
class EQAModelOutput(ModelOutput):
    loss=None
    logits=None
    ext_attention_mask = None
    ext_input_ids = None
    ext_hiddens = None
    ext_preds=None
    ext_lists=None

class TwoStageV2ModelOutput(ModelOutput):
    ext_loss: Optional[torch.FloatTensor] = None
    ext_logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class InfoModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    info_last_hidden_state: torch.FloatTensor = None
    info_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    info_attentions: Optional[Tuple[torch.FloatTensor]] = None

class InfoSeq2SeqModelOutput(ModelOutput):

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_info_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_info_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_info_attentions: Optional[Tuple[torch.FloatTensor]] = None


class LongformerSequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    logits_before_pad: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None


class TwoStageModelOutput(ModelOutput):
    """ Class for two-stage model """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    ext_input_ids: torch.LongTensor = None
    ext_preds: Optional[torch.LongTensor] = None
    #ext_logits: Optional[torch.Tensor] = None


class ExtractorOutput(ModelOutput):
    """ Class for Extractor output """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    logits_before_pad: torch.FloatTensor = None
    #encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    #encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    #encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    #past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    #decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    #decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    #cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

class SampleExtractorOutput(ModelOutput):
    """ Class for sample function in generation_with_grad """
    ext_logits: torch.FloatTensor = None
    ext_attention_mask: Optional[torch.Tensor] = None
    ext_input_ids: torch.LongTensor = None
    ext_preds: Optional[torch.LongTensor] = None
    #ext_scores: Optional[Tuple[torch.FloatTensor]] = None
    #ext_input_ids: Optional[torch.Tensor] = None

class SampleOutput(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ext_logits: torch.FloatTensor = None
    ext_attention_mask: Optional[torch.Tensor] = None
    ext_input_ids: torch.LongTensor = None


class GreedySearchOutput(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ext_logits: torch.FloatTensor = None
    ext_attention_mask: Optional[torch.Tensor] = None
    ext_input_ids: torch.LongTensor = None
