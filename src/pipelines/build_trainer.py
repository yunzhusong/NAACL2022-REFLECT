""" Builds trainer. """
import nltk
import numpy as np

from random import random
from datasets import load_metric
from transformers import DataCollatorForSeq2Seq

import pdb

def build_trainer(model_args, data_args, training_args, model, tokenizer, train_dataset,
                  eval_dataset, test_dataset):

    if training_args.output_extraction_results:
        ori_test_dataset = test_dataset["origin"] if training_args.do_predict else None
        ori_eval_dataset = eval_dataset["origin"] if training_args.do_eval else None
        eval_dataset = eval_dataset["processed"] if training_args.do_eval else None
    else:
        ori_test_dataset = None
        ori_eval_dataset = None

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    exp_id = random()

    metric = load_metric("rouge")
    metric_acc = load_metric("accuracy")
    metric_f1 = load_metric("f1")
    metric_recall = load_metric("recall")
    metric_precision = load_metric("precision")

    def _postprocess_text(texts):
        texts = [text.strip() for text in texts]
        texts = ["\n".join(nltk.sent_tokenize(text)) for text in texts]
        return texts

    def postprocess_text(preds, labels):
         preds = _postprocess_text(preds)
         labels = _postprocess_text(labels)
         return preds, labels

    def compute_metrics_ext(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds, preds_ext, index_ext = preds
        if isinstance(labels, tuple):
            labels, labels_ext = labels
            
        decoded_preds_ext = tokenizer.batch_decode(preds_ext, skip_special_tokens=True)

        if data_args.ignore_pad_token_for_loss:
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels,
                                                skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds_ext = _postprocess_text(decoded_preds_ext)
        decoded_labels = _postprocess_text(decoded_labels)

        # Extract a few results from ROUGE
        result_ext = metric.compute(predictions=decoded_preds_ext,
                                references=decoded_labels,
                                use_stemmer=True)
        result_ext = {"ext_"+key: value.mid.fmeasure * 100 for key, value in result_ext.items()}
        prediction_lens_ext = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds_ext]
        result_ext["ext_gen_len"] = np.mean(prediction_lens_ext)

        result = {}
        if preds is not None:
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_preds = _postprocess_text(decoded_preds)
            result = metric.compute(predictions=decoded_preds,
                                    references=decoded_labels,
                                    use_stemmer=True)
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)

        all_result = {**result, **result_ext}

        # The extraction performance
        if labels_ext is not None:
            padding_mask = labels_ext != -100
            _index_ext = index_ext[padding_mask]
            _labels_ext = labels_ext[padding_mask]

            acc= metric_acc.compute(predictions=_index_ext, references=_labels_ext)["accuracy"]
            f1 = metric_f1.compute(predictions=_index_ext, references=_labels_ext)["f1"]
            recall = metric_recall.compute(predictions=_index_ext, references=_labels_ext)["recall"]
            precision = metric_precision.compute(predictions=_index_ext, references=_labels_ext)["precision"]
            all_result["ext_acc"] = acc
            all_result["ext_f1"] = f1
            all_result["ext_recall"] = recall
            all_result["ext_precision"] = precision
            all_result["extraction_ratio"] = _index_ext.mean()

            # Print an example
            data_id = 1
            #print("\nGenerated Summary  :\n{}\n".format(decoded_preds[data_id]))
            #print("\nExtracted Sentences:\n{}\n".format(decoded_preds_ext[data_id].replace("\n", " ")))
            num_sent = sum(padding_mask[data_id])
            example_index = np.where(index_ext[data_id]==1)[0].tolist()
            example_label = np.where(labels_ext[data_id]==1)[0].tolist()
            print("({}/{}) Extract:".format(len(example_index), num_sent), example_index)
            print("({}/{}) Oracle :".format(len(example_label), num_sent), example_label)

        all_result = {k: round(v, 4) for k, v in all_result.items()}
        return all_result

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels,
                                                skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(
            decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds,
                                references=decoded_labels,
                                use_stemmer=True)

        # Extract a few results from ROUGE
        result = {
            key: value.mid.fmeasure * 100 for key, value in result.items()
        }

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        print(result)
        return result

    # Initialize our Trainer
    if "mle" == training_args.training_type:
        """ Train Abstractor MLE """
        from .trainer_mle import CustomSeq2SeqTrainer
        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            model_args=model_args, # NEW: for record
            data_args=data_args, #NEW: for record
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
            if training_args.predict_with_generate else None,
        )
    elif "ext_mle" == training_args.training_type:
        """ Train Extractor MLE """
        from .trainer_ext_mle import CustomTrainer
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            model_args=model_args, # NEW: for record
            data_args=data_args, #NEW: for record
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            ori_eval_dataset=ori_eval_dataset if training_args.do_eval else None, # NEW
            ori_test_dataset=ori_test_dataset if training_args.do_predict else None, # NEW: for
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_ext,
        )
    elif "ext_rl" == training_args.training_type:
        """ Train Extractor RL """
        from .trainer_ext_rl import CustomTrainer
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            model_args=model_args, # NEW: for record
            data_args=data_args, #NEW: for record
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            ori_eval_dataset=ori_eval_dataset if training_args.do_eval else None, # NEW
            ori_test_dataset=ori_test_dataset if training_args.do_predict else None, # NEW: for
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_ext,
        )

    return trainer
