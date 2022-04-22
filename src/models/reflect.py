import math
import nltk
import numpy as np
import copy

import torch
from torch import nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch_scatter import scatter_mean

from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
from transformers.models.bart.modeling_bart import BartPretrainedModel

from models.modeling_output import EQAModelOutput
from models.model_utils import SoftCrossEntropyLoss

# For rl
from torch.distributions.categorical import Categorical # For extract()

import pdb

class ReflectModel(BartPretrainedModel):
    def __init__(self, args, tokenizer, abstractor, extractor, section_extractor=None):
        super().__init__(abstractor.config)
       
        self.args = args
        self.tokenizer = tokenizer

        self.additional_input_kwargs = [
            "info_input_ids", "info_attention_mask", "info_sent_index", "num_sent", "ext_labels", "ext_scores",
            "gen_input_ids", "gen_sent_index"
        ]

        self.abstractor = abstractor
        self.num_labels = extractor.config.num_labels
        self.extractor = extractor
        self.section_extractor = section_extractor

        if args.score_regression:
            self.tr_loss_reg = torch.tensor(0.0).to(args.device)

        self.init_weights()


    def forward(
        self,
        info_input_ids=None,
        info_attention_mask=None,
        info_sent_index=None,
        ext_labels=None,
        ext_scores=None,
        gen_input_ids=None,
        gen_sent_index=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **inputs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        loss = None

        if self.args.data_level == "article":
            outputs = self.article_extract(
                info_input_ids=info_input_ids,
                info_attention_mask=info_attention_mask,
                info_sent_index=info_sent_index,
                ext_labels=ext_labels,
                ext_scores=ext_scores,
                gen_input_ids=gen_input_ids,
                gen_sent_index=gen_sent_index,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **inputs,
            )


        elif self.args.data_level == "section":
            outputs = self.section_extract(
                info_input_ids=info_input_ids,
                info_attention_mask=info_attention_mask,
                info_sent_index=info_sent_index,
                ext_labels=ext_labels,
                xt_scores=ext_scores,
                gen_input_ids=gen_input_ids,
                gen_sent_index=gen_sent_index,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **inputs,
            )

        elif self.args.data_level == "none":
            """ No sentence encoder """

            outputs = self.extract(
                info_input_ids=info_input_ids,
                info_attention_mask=info_attention_mask,
                info_sent_index=info_sent_index,
                ext_labels=ext_labels,
                ext_scores=ext_scores,
                gen_input_ids=gen_input_ids,
                gen_sent_index=gen_sent_index,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **inputs,
            )

        if not return_dict:
            output = (outputs.logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return outputs

    def article_extract(
        self,
        info_input_ids=None,
        info_attention_mask=None,
        info_sent_index=None,
        num_sent=None,
        ext_labels=None,
        ext_scores=None,
        gen_input_ids=None,
        gen_sent_index=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_loss=True,
        return_extraction_results=False,
        sample=False,
        **inputs,
        ):

        bs = len(info_input_ids)
        num_sents = num_sent.squeeze()

        # First-level encoding: extract sentence features
        batch_len = info_input_ids.shape[1]
        chunk_num = math.ceil(batch_len / 512)
        info_input_ids_chunks = torch.chunk(info_input_ids, chunk_num, dim=-1)
        info_attention_mask_chunks = torch.chunk(info_attention_mask, chunk_num, dim=-1)

        outputs_list = []
        for sec_input_ids, sec_attention_mask in zip(info_input_ids_chunks,
                                                     info_attention_mask_chunks):
            outputs = self.section_extractor.roberta(
                sec_input_ids,
                attention_mask=sec_attention_mask,
            )[0]
            outputs_list.append(outputs)
        outputs_cat = torch.cat(outputs_list, dim=1)
        qa_embeds = scatter_mean(outputs_cat, info_sent_index, dim=1)
        qa_attention_mask = scatter_mean(info_attention_mask, info_sent_index, dim=1)

        src_len = qa_embeds.shape[1] # bos + the number of article sentenences
        ref_len = 0

        max_gen_sent = 28
        max_sent = 512 - max_gen_sent

        # Do summary-reference
        if self.args.reference_extraction:
            gen_attention_mask = (gen_input_ids!=1).long()
            gen_embeds = self.section_extractor.roberta(
                gen_input_ids,
                attention_mask=gen_attention_mask,
            )[0]

            gen_embeds = scatter_mean(gen_embeds, gen_sent_index, dim=1)
            gen_embeds = torch.cat((gen_embeds[:,0:1], gen_embeds[:,3:], gen_embeds[:,2:3]), dim=1)
            gen_attention_mask = (gen_embeds[:,:,0]!=0).long()

            qa_embeds = torch.cat((gen_embeds, qa_embeds), dim=1)
            qa_attention_mask = torch.cat((gen_attention_mask, qa_attention_mask), dim=1).to(self.device)

            ref_len = gen_embeds.shape[1]

        # Second-level encoding
        ext_logits = self.extractor(
            inputs_embeds=qa_embeds,
            attention_mask=qa_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Only extract from the articles and exclude the first token (which is eos token)
        if self.args.score_regression:
            logits = ext_logits["cls"][:,ref_len+1:]
            logits_reg = ext_logits["reg"][:,ref_len+1:]
        else:
            logits = ext_logits[:,ref_len+1:]

        if return_extraction_results:

            if sample:
                m = Categorical(logits=logits)
                pred = m.sample() 
            else:
                m = None
                pred = torch.argmax(logits, dim=-1)

            for i, num_sent in enumerate(num_sents):
                pred[i,num_sent:] = 0
            preds = [pred, m]

            # ---
            # Whether should we give the sent representation or the token id as info information
            max_output_len = 1024
            ext_input_ids = torch.ones((bs, max_output_len), dtype=info_input_ids.dtype).to(self.device)
            ext_input_ids[:,0] = self.tokenizer.bos_token_id

            for b in range(bs):
                #indice = torch.where(debug_labels[b]==1)[0]
                indice = torch.where(pred[b]==1)[0]
                chosen_positions= torch.zeros(info_sent_index.shape[1], dtype=torch.long).to(self.device)
                for index in indice:
                    chosen_position = info_sent_index[b] == (index+1)
                    chosen_positions += chosen_position

                ext_ids = info_input_ids[b][chosen_positions.bool()][:max_output_len-2]
                ext_input_ids[b, 1:1+len(ext_ids)] = ext_ids
                ext_input_ids[b, 1+len(ext_ids)] = self.tokenizer.eos_token_id

            ext_attention_mask = (ext_input_ids!=self.tokenizer.pad_token_id).int()

        # Pad the logits by the first logits
        pad_len = max_sent - logits.shape[1]
        logits = torch.cat((logits, logits[:,:1,:].repeat(1, pad_len, 1)), dim=1)

        loss = None
        if return_loss and ext_labels is not None:
            if self.args.score_cls_weighting:
                loss_fct = SoftCrossEntropyLoss()
                loss = loss_fct(logits.reshape(-1, self.num_labels), ext_labels.reshape(-1), ext_scores.reshape(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.reshape(-1, self.num_labels), ext_labels.reshape(-1))
            
            if self.args.score_regression:
                logits_reg = torch.cat((logits_reg, logits_reg[:,:1,:].repeat(1, pad_len, 1)), dim=1)
                loss_fct_reg = MSELoss(reduction='none')
                loss_reg = loss_fct_reg(logits_reg.reshape(-1), ext_scores.reshape(-1))
                loss_mask = (ext_scores != loss_fct.ignore_index).reshape(-1).int()
                loss_reg = (loss_reg * loss_mask).sum()/loss_mask.sum()
                self.tr_loss_reg += loss_reg.item()
                loss += loss_reg 

        ext_outputs = EQAModelOutput(
            loss=loss,
            logits=logits,
            ext_attention_mask=ext_attention_mask if return_extraction_results else None,
            ext_input_ids=ext_input_ids if return_extraction_results else None,
            ext_preds=preds if return_extraction_results else None,
            #ext_lists=ext_lists,
        )
        return ext_outputs

    def section_extract(
        self,
        info_input_ids=None,
        info_attention_mask=None,
        info_sent_index=None,
        num_sent=None,
        ext_labels=None,
        ext_scores=None,
        gen_input_ids=None,
        gen_sent_index=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_loss=True,
        return_extraction_results=False,
        sample=False,
        **inputs,
        ):

        bs = len(info_input_ids)
        num_sents = num_sent.squeeze()

        # First-level encoding: extract sentence features
        outputs = self.section_extractor.roberta(
            info_input_ids,
            info_attention_mask,
        )[0]
        qa_embeds = scatter_mean(outputs, info_sent_index, dim=1)
        qa_attention_mask = scatter_mean(info_attention_mask, info_sent_index, dim=1)

        src_len = qa_embeds.shape[1] # bos + the number of article sentenences
        ref_len = 0

        max_gen_sent = 28
        max_sent = 512 - max_gen_sent

        # Do summary-reference
        if self.args.reference_extraction:
            gen_attention_mask = (gen_input_ids!=1).long()
            gen_embeds = self.section_extractor.roberta(
                gen_input_ids,
                attention_mask=gen_attention_mask,
            )[0]

            gen_embeds = scatter_mean(gen_embeds, gen_sent_index, dim=1)
            gen_embeds = torch.cat((gen_embeds[:,0:1], gen_embeds[:,3:], gen_embeds[:,2:3]), dim=1)
            gen_attention_mask = (gen_embeds[:,:,0]!=0).long()

            qa_embeds = torch.cat((gen_embeds, qa_embeds), dim=1)
            qa_attention_mask = torch.cat((gen_attention_mask, qa_attention_mask), dim=1).to(self.device)

            ref_len = gen_embeds.shape[1]

        # Second-level encoding
        ext_logits = self.section_extractor(
            encoder_outputs=[qa_embeds],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Only extract from the articles and exclude the first token (which is eos token)
        if self.args.score_regression:
            logits = ext_logits["cls"][:,ref_len+1:]
            logits_reg = ext_logits["reg"][:,ref_len+1:]
        else:
            logits = ext_logits[:,ref_len+1:]

        if return_extraction_results:

            if sample:
                m = Categorical(logits=logits)
                pred = m.sample() 
            else:
                m = None
                pred = torch.argmax(logits, dim=-1)

            for i, num_sent in enumerate(num_sents):
                pred[i,num_sent:] = 0
            preds = [pred, m]

            # ---
            # Whether should we give the sent representation or the token id as info information
            #if self.args.output_extraction_results:
            #    max_output_len = 2048
            #else:
            #    max_output_len = 1024
            max_output_len = 1024
            ext_input_ids = torch.ones((bs, max_output_len), dtype=info_input_ids.dtype).to(self.device)
            ext_input_ids[:,0] = self.tokenizer.bos_token_id

            for b in range(bs):
                #indice = torch.where(debug_labels[b]==1)[0]
                indice = torch.where(pred[b]==1)[0]
                chosen_positions= torch.zeros(info_sent_index.shape[1], dtype=torch.long).to(self.device)
                for index in indice:
                    chosen_position = info_sent_index[b] == (index+1)
                    chosen_positions += chosen_position

                ext_ids = info_input_ids[b][chosen_positions.bool()][:max_output_len-2]
                ext_input_ids[b, 1:1+len(ext_ids)] = ext_ids
                ext_input_ids[b, 1+len(ext_ids)] = self.tokenizer.eos_token_id

            ext_attention_mask = (ext_input_ids!=self.tokenizer.pad_token_id).int()

        # Pad the logits by the first logits
        pad_len = max_sent - logits.shape[1]
        logits = torch.cat((logits, logits[:,:1,:].repeat(1, pad_len, 1)), dim=1)

        loss = None
        if return_loss and ext_labels is not None:
            if self.args.score_cls_weighting:
                loss_fct = SoftCrossEntropyLoss()
                loss = loss_fct(logits.reshape(-1, self.num_labels), ext_labels.reshape(-1), ext_scores.reshape(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.reshape(-1, self.num_labels), ext_labels.reshape(-1))
            
            if self.args.score_regression:
                logits_reg = torch.cat((logits_reg, logits_reg[:,:1,:].repeat(1, pad_len, 1)), dim=1)
                loss_fct_reg = MSELoss(reduction='none')
                loss_reg = loss_fct_reg(logits_reg.reshape(-1), ext_scores.reshape(-1))
                loss_mask = (ext_scores != loss_fct.ignore_index).reshape(-1).int()
                loss_reg = (loss_reg * loss_mask).sum / loss_mask.sum()
                self.tr_loss_reg += loss_reg.item()
                loss += loss_reg

        ext_outputs = EQAModelOutput(
            loss=loss,
            logits=logits,
            ext_attention_mask=ext_attention_mask if return_extraction_results else None,
            ext_input_ids=ext_input_ids if return_extraction_results else None,
            ext_preds=preds if return_extraction_results else None,
        )
        return ext_outputs

    def extract(
        self,
        info_input_ids=None,
        info_attention_mask=None,
        info_sent_index=None,
        num_sent=None,
        ext_labels=None,
        ext_scores=None,
        gen_input_ids=None,
        gen_sent_index=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_loss=True,
        return_extraction_results=False,
        sample=False,
        **inputs,
        ):

        bs = len(info_input_ids)
        num_sents = num_sent.squeeze()

        #token_embeds = self.abstractor.model.shared(
        token_embeds = self.extractor.roberta.embeddings.word_embeddings(
            info_input_ids,
        )
        qa_embeds = scatter_mean(token_embeds, info_sent_index, dim=1)
        qa_attention_mask = scatter_mean(info_attention_mask, info_sent_index, dim=1)

        src_len = qa_embeds.shape[1] # bos + the number of article sentenences
        ref_len = 0

        max_gen_sent = 28
        max_sent = 512 - max_gen_sent

        if self.args.reference_extraction:

            gen_embeds = self.extractor.roberta.embeddings.word_embeddings(
            #gen_embeds = self.abstractor.model.shared(
                gen_input_ids,
            )
            gen_embeds = scatter_mean(gen_embeds, gen_sent_index, dim=1)
            gen_embeds = torch.cat((gen_embeds[:,0:1], gen_embeds[:,3:], gen_embeds[:,2:3]), dim=1)
            gen_attention_mask = (gen_embeds[:,:,0]!=0).long()

            qa_embeds = torch.cat((gen_embeds, qa_embeds), dim=1)
            qa_attention_mask = torch.cat((gen_attention_mask, qa_attention_mask), dim=1).to(self.device)

            ref_len = gen_embeds.shape[1]

        ext_logits = self.extractor(
            inputs_embeds=qa_embeds,
            attention_mask=qa_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # Only extract from the articles and exclude the first token (which is eos token)
        if self.args.score_regression:
            logits = ext_logits["cls"][:,ref_len+1:]
            logits_reg = ext_logits["reg"][:,ref_len+1:]
        else:
            logits = ext_logits[:,ref_len+1:]

        if return_extraction_results:

            if sample:
                m = Categorical(logits=logits)
                pred = m.sample() 
            else:
                m = None
                pred = torch.argmax(logits, dim=-1)

            for i, num_sent in enumerate(num_sents):
                pred[i,num_sent:] = 0
            #pred[ext_labels[:,:pred.shape[1]]==-100]=0
            preds = [pred, m]

            # NOTE: DE BUG
            #debug_labels = copy.deepcopy(ext_labels[:,:pred.shape[1]])
            #debug_labels[debug_labels==-100] = 0
            #preds = [debug_labels, m]

            # ---
            # Whether should we give the sent representation or the token id as info information
            #if self.args.output_extraction_results:
            #    max_output_len = 2048
            #else:
            #    max_output_len = 1024
            max_output_len = 1024
            ext_input_ids = torch.ones((bs, max_output_len), dtype=info_input_ids.dtype).to(self.device)
            ext_input_ids[:,0] = self.tokenizer.bos_token_id

            for b in range(bs):
                #indice = torch.where(debug_labels[b]==1)[0]
                indice = torch.where(pred[b]==1)[0]
                chosen_positions= torch.zeros(info_sent_index.shape[1], dtype=torch.long).to(self.device)
                for index in indice:
                    chosen_position = info_sent_index[b] == (index+1)
                    chosen_positions += chosen_position

                ext_ids = info_input_ids[b][chosen_positions.bool()][:max_output_len-2]
                ext_input_ids[b, 1:1+len(ext_ids)] = ext_ids
                ext_input_ids[b, 1+len(ext_ids)] = self.tokenizer.eos_token_id

            ext_attention_mask = (ext_input_ids!=self.tokenizer.pad_token_id).int()

        # Pad the logits by the first logits
        pad_len = max_sent - logits.shape[1]
        logits = torch.cat((logits, logits[:,:1,:].repeat(1, pad_len, 1)), dim=1)

        loss = None
        if return_loss and ext_labels is not None:
            if self.args.score_cls_weighting:
                loss_fct = SoftCrossEntropyLoss()
                loss = loss_fct(logits.reshape(-1, self.num_labels), ext_labels.reshape(-1), ext_scores.reshape(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.reshape(-1, self.num_labels), ext_labels.reshape(-1))
            
            if self.args.score_regression:
                logits_reg = torch.cat((logits_reg, logits_reg[:,:1,:].repeat(1, pad_len, 1)), dim=1)
                loss_fct_reg = MSELoss(reduction='none')
                loss_reg = loss_fct_reg(logits_reg.reshape(-1), ext_scores.reshape(-1))
                loss_mask = (ext_scores != loss_fct.ignore_index).reshape(-1).int()
                loss_reg = (loss_reg * loss_mask).sum() / loss_mask.sum()
                self.tr_loss_reg += loss_reg.item()
                loss += loss_reg

        ext_outputs = EQAModelOutput(
            loss=loss,
            logits=logits,
            ext_attention_mask=ext_attention_mask if return_extraction_results else None,
            ext_input_ids=ext_input_ids if return_extraction_results else None,
            ext_preds=preds if return_extraction_results else None,
        )
        return ext_outputs

    def get_encoder(self):
        return self.abstractor.model.encoder

    def get_decoder(self):
        return self.abstractor.model.decoder

class HierarchicalRobertaForTokenClassification(RobertaPreTrainedModel):
    """ The extractor of RERLECT"""
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.args = args

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        if args.score_regression:
            self.regressor = nn.Linear(config.hidden_size, 1)

        self.init_weights()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        encoder_outputs=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        sequence_output = encoder_outputs[0]

        if self.args.score_regression:
            logits = self.qa_outputs(sequence_output)
            logits_reg = self.regressor(sequence_output)
            return {"cls": logits, "reg": logits_reg}
        else:
            logits = self.qa_outputs(sequence_output)
            return logits

