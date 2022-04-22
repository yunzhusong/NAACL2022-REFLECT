""" Trainer for Transfer Learning

The code is developed based on:
    https://github.com/huggingface/transformers/blob/v4.4.2/src/transformers/trainer.py
"""
import os
import nltk
import shutil
from typing import Optional, Union, Dict, Any, Callable, Tuple, List

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataset import Dataset

#from transformers import Trainer
from transformers import Seq2SeqTrainer
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
from transformers.models.t5.modeling_t5 import T5LayerNorm
from transformers.utils import logging

from datasets import load_metric
from random import random
from others.arguments import ModelArguments, DataTrainingArguments

import pandas as pd
import ast
from itertools import chain

from transformers.models.bart.modeling_bart import _make_causal_mask, _expand_mask, shift_tokens_right
import pdb

## import for prediction_loop ()
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedTensorGatherer,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    get_parameter_names,
    nested_concat,
)
from transformers.file_utils import (
    WEIGHTS_NAME,
    is_torch_tpu_available,
)
import collections
from transformers.trainer_utils import (
    PredictionOutput,
    denumpify_detensorize,
)
##

logger = logging.get_logger(__name__)

class CustomTrainer(Seq2SeqTrainer):

    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module] = None,
        args: TrainingArguments = None,
        model_args: ModelArguments = None,
        data_args: DataTrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        ori_eval_dataset: Optional[Dataset] = None, # NEW
        ori_test_dataset: Optional[Dataset] = None, # NEW
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer,
                          torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        Seq2SeqTrainer.__init__(self, model, args, data_collator, train_dataset,
                                eval_dataset, tokenizer, model_init,
                                compute_metrics, callbacks, optimizers)

        self.model_args = model_args
        self.data_args = data_args

        # Make specified parameters trainable
        self._freeze_all_params(self.model)
        self._unfreeze_specified_params(self.model, train_only=args.train_only)

        # Show number of parameters
        all_param_num = sum([p.nelement() for p in self.model.parameters()])
        trainable_param_num = sum([
            p.nelement()
            for p in self.model.parameters()
            if p.requires_grad == True
        ])
        print(f"All parameters : {all_param_num}")
        print(f"Trainable parameters : {trainable_param_num}")

        # For best model saving
        self._ckpt_eval_loss = {}
        if self.args.save_model_accord_to_rouge:
            self._ckpt_eval_rouge = {}

        # Compute ROUGE
        #self.rouge = load_metric("rouge", experiment_id='{}'.format(random()))
        self.rouge = load_metric("rouge")

	# Metric
        exp_id = random()
        self.metric = load_metric("rouge", experiment_id=exp_id) if args.predict_with_generate else None
        self.metric_acc = load_metric("accuracy", experiment_id=exp_id) if "ext" in args.training_type else None
        self.metric_f1 = load_metric("f1", experiment_id=exp_id) if "ext" in args.training_type else None
        self.metric_recall = load_metric("recall", experiment_id=exp_id) if "ext" in args.training_type else None
        self.metric_precision = load_metric("precision", experiment_id=exp_id) if "ext" in args.training_type else None

        self.ori_eval_dataset = ori_eval_dataset
        self.ori_test_dataset = ori_test_dataset

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch):
        """
        Modification:
            - specify max_length argument in self.evaluate()
            - record current eval loss/rouge for best model saving
        """
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar /
                (self.state.global_step - self._globalstep_last_logged), 4)

            logs["learning_rate"] = self._get_learning_rate()

            # NEW: record additional loss
            if self.args.score_regression:
                tr_loss_reg_scalar = self.model.tr_loss_reg.item()
                # reset tr_loss_reg to zero
                self.model.tr_loss_reg -= self.model.tr_loss_reg

                logs["loss_reg"] = round(
                    tr_loss_reg_scalar /
                    (self.state.global_step - self._globalstep_last_logged), 4)

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            self.log({**logs, **self.rl_logs})

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(
                max_length=self.args.train_val_max_target_length
            )  # NOTE: no max length for Trainer
            self._report_to_hp_search(trial, epoch, metrics)

            # NEW: record metric
            if self.args.save_model_accord_to_rouge:
                self._cur_eval_rouge = metrics['eval_rouge1']
            self._cur_eval_loss = metrics['eval_loss']

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control)

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        """
        Modification:
            - record eval loss/rouge and maintain best model

        NOTE: to make this function works properly,
              the save_steps should be multiples of evaluation_steps
        """
        # NEW
        if self.args.eval_steps != self.args.save_steps:
            raise Exception(
                "To properly store best models, please make sure eval_steps equals to save_steps."
            )

        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Sort according to evaluation steps
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime,
                                                      output_dir=output_dir)

        # NEW: record the eval metric for the last checkpoint
        self._ckpt_eval_loss[checkpoints_sorted[-1]] = self._cur_eval_loss
        if self.args.save_model_accord_to_rouge:
            self._ckpt_eval_rouge[checkpoints_sorted[-1]] = self._cur_eval_rouge

        # Check if we should delete older checkpoint(s)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # NEW: sort according to metrics (descending for loss)
        checkpoints_sorted = [
            k for k, v in sorted(
                self._ckpt_eval_loss.items(), key=lambda x: x[1], reverse=True)
        ]

        number_of_checkpoints_to_delete = max(
            0,
            len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:
                                                       number_of_checkpoints_to_delete]

        # NEW: sort according to metrics (ascending for rouge)
        if self.args.save_model_accord_to_rouge:
            checkpoints_sorted_rouge = [
                k for k, v in sorted(self._ckpt_eval_rouge.items(),
                                     key=lambda x: x[1],
                                     reverse=False)
            ]
            checkpoints_to_be_deleted_rouge = checkpoints_sorted_rouge[:
                                                                       number_of_checkpoints_to_delete]
            # Only delete the intersect checkpoints
            checkpoints_to_be_deleted = list(
                set(checkpoints_to_be_deleted).intersection(
                    set(checkpoints_to_be_deleted_rouge)))

        for checkpoint in checkpoints_to_be_deleted:
            logger.info(
                "Deleting older checkpoint [{}] due to args.save_total_limit".
                format(checkpoint))
            del self._ckpt_eval_loss[checkpoint]  # NEW: remove the delted ckpt
            if self.args.save_model_accord_to_rouge:
                del self._ckpt_eval_rouge[checkpoint]  # NEW: remove the delted ckpt
            shutil.rmtree(checkpoint)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Modification:
            - Also record model_args and data_args
        """
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(output_dir,
                                                         state_dict=state_dict)
            else:
                logger.info(
                    "Trainer.model is not a `PreTrainedModel`, only saving its state dict."
                )
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        torch.save(self.model_args, os.path.join(output_dir,
                                                 "model_args.bin"))  # NEW
        torch.save(self.data_args, os.path.join(output_dir,
                                                "data_args.bin"))  # NEW

    def _freeze_all_params(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def _unfreeze_specified_params(self, model, train_only=None):
        if train_only is not None:
            print("[Warning] Not train the section extractor")
            names = train_only.split()
            for train_name in names:
                for name, sub_module in self.model.named_modules():
                    if train_name in name and "section" not in name:
                        for param in sub_module.parameters():
                            param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = True

    def prediction_loop(
        self,
        dataloader,
        description,
        prediction_loss_only= None,
        ignore_keys: Optional = None,
        metric_key_prefix = "eval",
    ):
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )
        predict_with_generate_abs = self.args.predict_with_generate_abs

        if self.args.deepspeed and not self.args.do_train:
            # no harm, but flagging to the user that deepspeed config is ignored for eval
            # flagging only for when --do_train wasn't passed as only then it's redundant
            logger.info("Detected the deepspeed argument but it will not be used for evaluation")

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, half it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Batch size = {batch_size}")
        losses_host: torch.Tensor = None
        preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None
        preds_ext_host: Union[torch.Tensor, List[torch.Tensor]] = None # NEW
        index_ext_host: Union[torch.Tensor, List[torch.Tensor]] = None # NEW
        labels_ext_host: Union[torch.Tensor, List[torch.Tensor]] = None # NEW

        world_size = max(1, self.args.world_size)

        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
        if not prediction_loss_only:
            # The actual number of eval_sample can be greater than num_examples in distributed settings (when we pass
            # a batch size to the sampler)
            make_multiple_of = None
            if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, SequentialDistributedSampler):
                make_multiple_of = dataloader.sampler.batch_size
            if predict_with_generate_abs:
                preds_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
            labels_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
            preds_ext_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
            index_ext_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
            labels_ext_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)

        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        for step, inputs in enumerate(dataloader):
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            if loss is not None:
                losses = loss.repeat(batch_size)
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                if predict_with_generate_abs:
                    preds_host = logits[0] if preds_host is None else nested_concat(preds_host, logits[0], padding_index=-100)
                preds_ext_host = logits[1] if preds_ext_host is None else nested_concat(preds_ext_host, logits[1], padding_index=-100)
                index_ext_host = logits[2] if index_ext_host is None else nested_concat(index_ext_host, logits[2], padding_index=-100)
            if labels is not None:
                labels_host = labels[0] if labels_host is None else nested_concat(labels_host, labels[0], padding_index=-100)
                labels_ext_host = labels[1] if labels_ext_host is None else nested_concat(labels_ext_host, labels[1], padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
                if not prediction_loss_only:
                    if predict_with_generate_abs:
                        preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
                    labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))
                    preds_ext_gatherer.add_arrays(self._gather_and_numpify(preds_ext_host, "eval_preds_ext"))
                    index_ext_gatherer.add_arrays(self._gather_and_numpify(index_ext_host, "eval_index_ext"))
                    labels_ext_gatherer.add_arrays(self._gather_and_numpify(labels_ext_host, "eval_label_ext"))

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None
                preds_ext_host, index_ext_host, labels_ext_host = None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
        if not prediction_loss_only:
            if predict_with_generate_abs:
                preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
            labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))
            preds_ext_gatherer.add_arrays(self._gather_and_numpify(preds_ext_host, "eval_preds_ext"))
            index_ext_gatherer.add_arrays(self._gather_and_numpify(index_ext_host, "eval_index_ext"))
            labels_ext_gatherer.add_arrays(self._gather_and_numpify(labels_ext_host, "eval_label_ext"))

        preds = preds_gatherer.finalize() if not prediction_loss_only and predict_with_generate_abs else None
        label_ids = labels_gatherer.finalize() if not prediction_loss_only else None
        preds_ext = preds_ext_gatherer.finalize() if not prediction_loss_only else None
        index_ext = index_ext_gatherer.finalize() if not prediction_loss_only else None
        label_ext = labels_ext_gatherer.finalize() if not prediction_loss_only else None
        eval_loss = eval_losses_gatherer.finalize()

        #if self.compute_metrics is not None and preds is not None and label_ids is not None:
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(
                EvalPrediction(predictions=(preds, preds_ext, index_ext), 
                                            label_ids=(label_ids, label_ext)))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        torch.cuda.empty_cache()

        #if self.args.output_extraction_results:
        #    return PredictionOutput(predictions=(preds, preds_ext), label_ids=label_ids, metrics=metrics)
        if self.args.output_extraction_results:
            if description == "Evaluation":
                ori_dataset = self.ori_eval_dataset
            elif description == "Prediction":
                ori_dataset = self.ori_test_dataset
            if self.data_args.text_column:
                articles = [ast.literal_eval(a) for a in ori_dataset[self.data_args.text_column]]
            else:
                articles = [ast.literal_eval(a) for a in ori_dataset['document']]

            padding_mask = label_ext != -100
            predictions = (index_ext == 1)*padding_mask
            ext_articles = []
            for prediction, article in zip(predictions, articles):
                if self.args.multi_documents:
                    article = list(chain.from_iterable(article))
                ext_articles.append(" ".join(
                    [s for p, s in zip(prediction, article) if p==1]))

            if self.data_args.summary_column:
                summary = ori_dataset[self.data_args.summary_column][:len(articles)]
            else:
                summary = ori_dataset["summary"][:len(articles)]
            extraction_results = pd.DataFrame({
                "ext_article": ext_articles,
                "summary": summary,
            })
            return PredictionOutput(predictions=(preds, extraction_results), label_ids=label_ids, metrics=metrics)


        else:
            return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        NOTE: When there are additional input args to model. We should modify this function.
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_ext_labels = "ext_labels" in inputs
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # NEW: for additional input 
        ext_kwargs = {}
        for k in self.model.additional_input_kwargs:
            ext_kwargs[k] = inputs[k] if k in inputs.keys() else None

        with torch.no_grad():
            ext_outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                sample=False,
                return_extraction_results=True,
                **ext_kwargs,
            )

        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.abstractor.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.abstractor.config.num_beams,
        }

        if self.args.predict_with_generate_abs:
            generated_tokens = self.model.abstractor.generate(
                input_ids=ext_outputs.ext_input_ids,
                attention_mask=ext_outputs.ext_attention_mask,
                **gen_kwargs,
            )

            # in case the batch is shorter than max length, the output should be padded
            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        else:
            generated_tokens=None

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_ext_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None
            #loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

	# Evaluate the extraction results
        ext_preds = ext_outputs.ext_preds[0]
        if has_ext_labels:
            ext_labels = inputs["ext_labels"][:,:ext_preds.shape[1]]
        else:
            ext_labels = None

        return (loss, (generated_tokens, ext_outputs.ext_input_ids, ext_preds), (labels, ext_labels))

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        self.rl_logs = {}
        self.print_result_in_prediction_step = True

        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs_spl = model(**inputs, return_loss=self.args.use_mixer_loss, sample=True, return_extraction_results=True)

        with torch.no_grad():
            outputs_gdy = model(**inputs, return_loss=False, sample=False, return_extraction_results=True)

            gen_kwargs = {
                "max_length": 256, # 256
                "num_beams": 1,
                "output_scores": False,
                "return_dict_in_generate": True,
                "early_stopping": True,
            }
            generation_spl = self.model.abstractor.generate(
                input_ids=outputs_spl['ext_input_ids'],
                attention_mask=outputs_spl['ext_attention_mask'],
                **gen_kwargs,
            )
            generation_gdy = self.model.abstractor.generate(
                input_ids=outputs_gdy['ext_input_ids'],
                attention_mask=outputs_gdy['ext_attention_mask'],
                **gen_kwargs,
            )

            indices_spl = generation_spl['sequences']
            indices_gdy = generation_gdy['sequences']

            gold_summs = self.tokenizer.batch_decode(
                labels * (labels >= 0), skip_special_tokens=True)

            rouge_type, rouge_score = "rougeL", "f"
            rewards_gdy = self.get_reward_by_rouge(indices_gdy, gold_summs, rouge_type=rouge_type, score=rouge_score)
            rewards_spl = self.get_reward_by_rouge(indices_spl, gold_summs, rouge_type=rouge_type, score=rouge_score)

            advantages = (rewards_spl - rewards_gdy).reshape(-1,1)

            self.rl_logs["advantage"] = round(advantages.mean().item(), 4)
            self.rl_logs["reward_spl_{}{}".format(rouge_type, rouge_score)] = round(rewards_spl.mean().item(), 4)
            self.rl_logs["reward_gdy_{}{}".format(rouge_type, rouge_score)] = round(rewards_gdy.mean().item(), 4)

            if self.args.use_ext_reward:
                rouge_type, rouge_score = "rouge1", "r"
                rewards_ext_gdy = self.get_reward_by_rouge(outputs_gdy["ext_input_ids"], gold_summs, rouge_type=rouge_type, score=rouge_score)
                rewards_ext_spl = self.get_reward_by_rouge(outputs_spl["ext_input_ids"], gold_summs, rouge_type=rouge_type, score=rouge_score)
                advantages_ext = (rewards_ext_spl - rewards_ext_gdy).reshape(-1,1)

                self.rl_logs["advantage_ext"] = round(advantages_ext.mean().item(), 4)
                self.rl_logs["reward_ext_spl_{}{}".format(rouge_type, rouge_score)] = round(rewards_ext_spl.mean().item(), 4)
                self.rl_logs["reward_ext_gdy_{}{}".format(rouge_type, rouge_score)] = round(rewards_ext_gdy.mean().item(), 4)

                advantages = 0.5*advantages + 0.5*advantages_ext

            """
            decoder_input_ids = shift_tokens_right(
                labels, self.model.config.pad_token_id, self.model.config.decoder_start_token_id
                )
            inputs["decoder_input_ids"] = decoder_input_ids

            abs_logits_spl = self.model.abstractor(
                input_ids=outputs_spl['ext_input_ids'],
                attention_mask=outputs_spl['ext_attention_mask'],
                decoder_input_ids=inputs['decoder_input_ids'],
            )['logits']
            abs_logits_gdy = self.model.abstractor(
                input_ids=outputs_gdy['ext_input_ids'],
                attention_mask=outputs_gdy['ext_attention_mask'],
                decoder_input_ids=inputs['decoder_input_ids'],
            )['logits']

            fct = CrossEntropyLoss(reduction='none')
            abs_loss_spl = fct(abs_logits_spl.reshape(-1, self.tokenizer.vocab_size), labels.reshape(-1)).reshape(labels.shape)
            abs_loss_gdy = fct(abs_logits_gdy.reshape(-1, self.tokenizer.vocab_size), labels.reshape(-1)).reshape(labels.shape)

            abs_loss_spl = torch.mean(abs_loss_spl, dim=1)
            abs_loss_gdy = torch.mean(abs_loss_gdy, dim=1)

            # The lower loss, the better sample. All action share the same advantage
            advantages = (abs_loss_gdy - abs_loss_spl).reshape(-1,1)
            """
        # if the output ext_preds is in one dimension, then repeat the advantages
        action, m = outputs_spl['ext_preds']
        ext_labels = inputs['ext_labels']
        log_probs = m.log_prob(action)
        action_mask = (ext_labels!=-100)[:,:action.shape[1]]

        rl_loss = 0
        #if self.args.use_sel_reward:
        #    selection_num = torch.sum(action, dim=1).reshape(-1,1) + 1
        #    rl_loss_sel = torch.sum(-log_probs*(rewards_spl.reshape(-1,1)*0.1)/selection_num * action_mask)
        #    rl_loss += rl_loss_sel
        #    self.rl_logs["sel_loss"] = round(rl_loss_sel.item(), 4)

        if not self.args.update_full_action:
            action_gdy = outputs_gdy["ext_preds"][0]
            diff = action!=action_gdy
            action_mask = action_mask * diff

        action_num = torch.sum(action_mask, dim=1).reshape(-1,1) + 1
        rl_loss_adv = torch.sum(-log_probs*(advantages)/action_num * action_mask)
        rl_loss += rl_loss_adv

        self.rl_logs["adv_loss"] = round(rl_loss_adv.item(), 4)
        self.rl_logs["rl_loss"] = round(rl_loss.item(), 4)

        if self.args.use_mixer_loss:
            mle_loss = outputs_spl.loss
            #mle_loss = mle_loss.reshape(ext_labels.shape) # NOTE: if consider each action
            #loss = rl_loss + self.args.mixer_weight*mle_loss
            loss = rl_loss + mle_loss/(1+self.state.epoch)
            self.rl_logs["mle_loss"] = round(mle_loss.item(), 4)

        else:
            loss = rl_loss

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # extraction loss
        outputs = {
            "loss": loss,
            "logits": indices_spl,
            #"logits": outputs_spl['logits'],
        }


        return (loss, outputs) if return_outputs else loss

    def get_reward_by_rouge(self, indices, target_summaries, rouge_type='rougeL', score="f"):
        """ Calculate the rouge score"""
        summaries = self.tokenizer.batch_decode(indices, skip_special_tokens=True)
        if score == "f":
            rewards = [r.fmeasure for r in self.rouge.compute(predictions=summaries, references=target_summaries,
                                            use_agregator=False, rouge_types=[rouge_type])[rouge_type]]
        elif score == "r":
            rewards = [r.recall for r in self.rouge.compute(predictions=summaries, references=target_summaries,
                                    use_agregator=False, rouge_types=[rouge_type])[rouge_type]]
        elif score == "p":
            rewards = [r.precision for r in self.rouge.compute(predictions=summaries, references=target_summaries,
                                    use_agregator=False, rouge_types=[rouge_type])[rouge_type]]
        

        return torch.tensor(rewards).to(self.args.device)

    def get_extraction_performance(self, preds, labels):

        result = self.metric_acc.compute(predictions=preds, references=labels)
        f1 = self.metric_f1.compute(predictions=preds, references=labels)["f1"]
        recall = self.metric_recall.compute(predictions=preds, references=labels)["recall"]
        precision = self.metric_precision.compute(predictions=preds, references=labels)["precision"]

        result["f1"] = f1
        result["recall"] = recall
        result["precision"] = precision
        result["extraction_ratio"] = torch.sum(preds)/len(preds)

        result = {k: round(v, 4) for k, v in result.items()}
        return result

