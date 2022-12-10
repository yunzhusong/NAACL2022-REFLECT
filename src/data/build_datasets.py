""" Builds datasets. """
import logging
import os
import ast
import nltk
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from data.utils import doc_truncation
from tqdm import tqdm
from itertools import chain

tqdm.pandas()
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=False)

summarization_name_mapping = {
    "multi_news": ("document", "summary"),
    "xscience": ("document", "related_work"),
    "wiki_cat_sum_animal": ("paragraphs", "summary"),
    "wiki_cat_sum_company": ("paragraphs", "summary"),
    "wiki_cat_sum_film": ("paragraphs", "summary"),
}

summarization_length_mapping = {
    "multi_news": (1024, 512),
    "xscience": (1024, 256),
    "wiki_cat_sum_animal": (1024, 128),
    "wiki_cat_sum_company": (1024, 128),
    "wiki_cat_sum_film": (1024, 128),
}

summarization_own_file_mapping = {
    "multi_news":
        "../datasets/ext_oracle/multi_news",
    "xscience":
        "../datasets/ext_oracle/xscience",
    "wiki_cat_sum_animal":
        "../datasets/ext_oracle/wikicatsum/animal",
    "wiki_cat_sum_company":
        "../datasets/ext_oracle/wikicatsum/company",
    "wiki_cat_sum_film":
        "../datasets/ext_oracle/wikicatsum/film",
}

logger = logging.getLogger(__name__)

def build_datasets(data_args,
                   training_args,
                   tokenizer,
                   ):

    data_dir = summarization_own_file_mapping.get(data_args.dataset_name, None)
    data_files = {}

    # Use the data_args.train_file/data_args.valudation_file/data_args.test_file if being assigned, 
    # otherwise, load from the data_dir
    extension = "csv"

    if training_args.do_train:
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        else:
            data_files["train"] = os.path.join(data_dir, "train.csv")
            extension = "csv"

    if training_args.do_eval:
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        else:
            data_files["validation"] = os.path.join(data_dir, "validation.csv")
            extension = "csv"

    if training_args.do_predict:
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        else:
            data_files["test"] = os.path.join(data_dir, "test.csv")
            extension = "csv"

    datasets = load_dataset(extension, data_files=data_files)

    # NEW: shuffle datasets before select
    if data_args.shuffle_before_select:
        datasets = datasets.shuffle(seed=0)

    # Take column names from datasets
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        raise "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        return

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name,
                                                     None)
    if data_args.text_column is None:
        text_column = dataset_columns[
            0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[
            1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )
    # NEW
    summary_ext_column = data_args.summary_ext_column
    if data_args.summary_ext_column is not None:
        summary_ext_column = summary_column + "_ext"

    # NOTE: Get max source length and max target length for the corpus
    max_source_length, max_target_length = summarization_length_mapping.get(
        data_args.dataset_name, None)
    if data_args.max_source_length is not None:
        max_source_length = data_args.max_source_length

    data_args.max_target_length = max_target_length
    data_args.val_max_target_length = max_target_length
    training_args.train_val_max_target_length = max_target_length

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    padding = "max_length" if data_args.pad_to_max_length else False

    # Arguments for the document-wise truncation function
    max_full_length = 6144 # max length for full article
    per_doc_length = 1024-2 # max length for full document
    logging.info("Max length for input/document:{}".format(max_full_length))

    # Arugments for the extraction task
    max_gen_sent = 28
    max_ext_length = 512 - max_gen_sent # save place for the generated summary

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id

    np_zero, np_one = np.array([0]), np.array([1])
    np_eos, np_bos, np_pad = np.array([eos]), np.array([bos]), np.array([pad])

    def _tokenize(data):
        """
        Use the outside variables:
            max_source_length
        """

        num_doc = 0
        total_num_tokens = 0
        tokenized_documents = []
        documents = data["article"]

        for i, document in enumerate(documents):
            if len(document)==0:
                document = ['  ']
            sents = document
            sents = [' '+sent for sent in sents]
            if i == 0:
                sents[0] = sents[0].strip()

            if document == '':
                continue

            inputs = tokenizer(sents, add_special_tokens=False)
            total_num_tokens += sum([len(ids) for ids in inputs['input_ids']])
            tokenized_documents.append(inputs)

            num_doc += 1

        input_ids, attention_mask = [], []
        per_doc_len = int((max_source_length-2) / num_doc) # max len for each document

        need_trun = True if total_num_tokens>max_source_length-2 else False
        for i, tokenized_document in enumerate(tokenized_documents):
            sents = tokenized_document['input_ids']
            masks = tokenized_document['attention_mask']

            input_ids_per_doc = list(chain.from_iterable(sents))
            attention_mask_per_doc = list(chain.from_iterable(masks))

            # For truncated article (control by per_doc_len).
            input_ids.append(input_ids_per_doc[:per_doc_len] if need_trun else input_ids_per_doc)
            attention_mask.append(attention_mask_per_doc[:per_doc_len] if need_trun else attention_mask_per_doc)

        # Article Fragment -------------------------------------|
        # Concatenate and pad to max_source_length
        _input_ids = list(chain.from_iterable(input_ids))
        _attention_mask = list(chain.from_iterable(attention_mask))

        source_length = max_source_length-2 # -2 for adding special tokens
        pad_len = max(0, source_length-len(_input_ids))
        _input_ids = [bos] + _input_ids[:source_length] + [eos] + [pad] * pad_len
        _attention_mask = [1] + _attention_mask[:source_length] + [1] + [0] * pad_len

        # ------------------------------------------------------|

        data["input_ids"] = _input_ids
        data["attention_mask"] = _attention_mask
        return data

    def _tokenize_and_build_sent_index_dense(data):
        """
        Use the outside variables:
            max_full_length, max_ext_length, max_source_length
            np_zero, np_one, np_eos, np_bos, np_pad
        """

        ext_sents = [i.strip() for i in data["summary_ext"]]
        #ext_sents = data["summary_ext"]
        #assert len(ext_sents)==len(data["summary_ext_idx"].split())

        target_index = []
        cnt = 0 # NOTE: the sentence index for target extraction. count from 0.

        num_doc = 0
        total_num_tokens = 0
        tokenized_documents = []
        documents = data["article"]

        #if len(documents) == 0:
        #    documents = [[""]]
        #    data["article"] = [["empty"]]
        #    data["summary_ext"] = [["empty"]]

        for i, document in enumerate(documents):
            #sents = nltk.sent_tokenize(document)
            sents = document
            sents = [' '+sent for sent in sents]
            if i == 0:
                sents[0] = sents[0].strip()

            if document == '':
                continue

            inputs = tokenizer(sents, add_special_tokens=False)
            total_num_tokens += sum([len(ids) for ids in inputs['input_ids']])
            tokenized_documents.append(inputs)

            if data["summary_ext"] is not None:
                # Obtain the sentence index for target extraction.
                for i, sent in enumerate(sents):
                    if sent.strip() in ext_sents:
                        ext_sents.remove(sent.strip())
                        target_index.append(i+cnt)
                cnt += len(sents) # the next document count from "cnt"
            num_doc += 1

        if data["summary_ext"] is not None:
            assert len(ext_sents)==0

        num_sent = 0
        input_ids, attention_mask = [], []
        sent_index_f, input_ids_f, attention_mask_f = [], [], [] # for ext-abs
        per_doc_len = int((max_source_length-2) / num_doc) # max len for each document

        need_trun = True if total_num_tokens>max_source_length-2 else False
        for i, tokenized_document in enumerate(tokenized_documents):
            sents = tokenized_document['input_ids']
            masks = tokenized_document['attention_mask']
            # NOTE: the sentence index count from 1. We save index 0 for eos, bos and pad tokens.
            index = [[j+1+num_sent]*len(tokens) for j, tokens in enumerate(sents)]

            input_ids_per_doc = list(chain.from_iterable(sents))
            attention_mask_per_doc = list(chain.from_iterable(masks))
            sent_index_per_doc = list(chain.from_iterable(index))

            # For truncated article (control by per_doc_len).
            input_ids.append(input_ids_per_doc[:per_doc_len] if need_trun else input_ids_per_doc)
            attention_mask.append(attention_mask_per_doc[:per_doc_len] if need_trun else attention_mask_per_doc)

            # The sentence number under the maximum document length.
            num_sent = sent_index_per_doc[-1]

            sent_index_f.append(sent_index_per_doc)
            input_ids_f.append(input_ids_per_doc)
            attention_mask_f.append(attention_mask_per_doc)
            
        # Article Fragment -------------------------------------|
        # Concatenate and pad to max_source_length
        _input_ids = list(chain.from_iterable(input_ids))
        _attention_mask = list(chain.from_iterable(attention_mask))

        source_length = max_source_length-2 # -2 for adding special tokens
        pad_len = max(0, source_length-len(_input_ids))
        _input_ids = [bos] + _input_ids[:source_length] + [eos] + [pad] * pad_len
        _attention_mask = [1] + _attention_mask[:source_length] + [1] + [0] * pad_len

        # Ext-Abs Preparation ---------------------------------|
        # Concatenate for full article.
        _sent_index_f = np.array(list(chain.from_iterable(sent_index_f)))
        _input_ids_f = np.array(list(chain.from_iterable(input_ids_f)))
        _attention_mask_f = np.array(list(chain.from_iterable(attention_mask_f)))

        if data["summary_ext"] is not None:
            _target_index = torch.tensor(target_index, dtype=torch.long) # Use tensor to cooperate the _scatter fn

        # Trauncate sentences to max_ext_length.
        ext_length = max_ext_length-3
        if num_sent > ext_length:
            num_sent = ext_length
            valid_sent_index = _sent_index_f<=ext_length
            _sent_index_f = _sent_index_f[valid_sent_index]
            _input_ids_f = _input_ids_f[valid_sent_index]
            _attention_mask_f = _attention_mask_f[valid_sent_index]
            if data["summary_ext"] is not None:
                _target_index = _target_index[_target_index<ext_length]

        # Add special tokens.
        full_length = max_full_length-2
        num_sent = _sent_index_f[:full_length][-1]
        _sent_index_f = np.concatenate((np_zero, _sent_index_f[:full_length], np.array([num_sent+1])))
        _input_ids_f = np.concatenate((np_eos, _input_ids_f[:full_length], np_eos)) # NOTE: 2 eos tokens
        _attention_mask_f = np.concatenate((np_one, _attention_mask_f[:full_length], np_one))


        # Padding to max_full_length
        pad_len = max(0, max_full_length-len(_sent_index_f))
        if pad_len > 0:
            _sent_index_f = np.concatenate((_sent_index_f, np.array([num_sent+2]).repeat(pad_len)))
            _input_ids_f = np.concatenate((_input_ids_f, np_pad.repeat(pad_len)))
            _attention_mask_f = np.concatenate((_attention_mask_f, np_zero.repeat(pad_len)))

        # Truncate and build binary label for extraction labels.
        # NOTE: the extraction labels indicate which sentence should be selected. 
        # Not consider the speical tokens.
        if data["summary_ext"] is not None:
            _target_onehot = torch.zeros(max_ext_length, dtype=torch.long)
            _target_onehot.scatter_(0, _target_index, 1)
            _target_onehot[num_sent:] = -100
            data["ext_labels"] = _target_onehot

        # ------------------------------------------------------|
        if data["art_rouges"] is not None:
            _ext_scores = torch.ones(max_ext_length, dtype=torch.float) * -100
            _ext_scores[:num_sent] = torch.tensor(data["art_rouges"][:num_sent])
            data["ext_scores"] = _ext_scores

        # ------------------------------------------------------|
        if data["summary_gen"] is not None:
            # Prepare for reference extraction
            gen_summary = data["summary_gen"]
            gen_summary = '</s> '.join(nltk.sent_tokenize(gen_summary))

            gen_inputs = tokenizer(
                gen_summary,
                max_length=384,
                padding="max_length",
                truncation=True,
                return_special_tokens_mask=True,
                return_tensors="pt"
            )
            gen_input_ids = gen_inputs["input_ids"]
            special_tokens_mask = gen_inputs["special_tokens_mask"]
            sent_end = torch.where(gen_input_ids==2, 1, 0)
            scatter_mask = torch.cumsum(sent_end, dim=1) + 3 # make room for special tokens
            scatter_mask = torch.where(scatter_mask>=max_gen_sent, max_gen_sent-1, scatter_mask)
            scatter_mask = torch.where(gen_input_ids==2, 2, scatter_mask) # all eos tokens will be scatter mean
            scatter_mask = torch.where(gen_input_ids==1, 1, scatter_mask) # all pad tokens will be scatter mean
            scatter_mask = torch.where(gen_input_ids==0, 0, scatter_mask) # all bos tokens will be scatter mean
            data["gen_input_ids"] = gen_input_ids.reshape(-1)
            data["gen_sent_index"] = scatter_mask.reshape(-1)

        # ------------------------------------------------------|
        data["input_ids"] = _input_ids
        data["attention_mask"] = _attention_mask
        data["info_sent_index"] = _sent_index_f
        data["info_input_ids"] = _input_ids_f
        data["info_attention_mask"] = _attention_mask_f
        data["num_sent"] = [num_sent]

        return data

    def preprocess_function(examples):
        inputs = examples[text_column]

        if data_args.data_preprocess != "raw":
            inputs = [inp if inp else "['we study']" for inp in inputs] #handle empty data
            if inputs[0][0] == '[':
                if data_args.text_column == "{}_ext".format(summary_column):
                    inputs = [[ast.literal_eval(i)] for i in inputs]
                
                elif not training_args.multi_documents:
                    inputs = [[ast.literal_eval(i)] for i in inputs]

                else:
                    inputs = [ast.literal_eval(i) for i in inputs]

                # TODO: add another [] for one_doc
        else:
            inputs = [inp if inp else " " for inp in inputs]
        
        if summary_column is not None:
            targets = examples[summary_column]
        else:
            targets = None

        if summary_ext_column is not None:
            targets_ext = examples[summary_ext_column]
        else:
            targets_ext = None

        # Truncate data in document-wise style
        if data_args.data_preprocess == "doc_trun_and_build_sent_index":
            """ Preprocess for extractor by document truncation and build sentence index"""
            df = pd.DataFrame({"article": inputs,
                               "summary_ext": None,
                               "summary_ext_idx": None,
                               "art_rouges": None,
                               "summary_gen": None,
                              })
            if data_args.summary_ext_column is not None:
                targets_ext = [ast.literal_eval(i) for i in targets_ext]
                df["summary_ext"] = targets_ext
                df["summary_ext_idx"] = examples["{}_ext_idx".format(summary_column)]

            if training_args.score_cls_weighting:
                art_rouges = [ast.literal_eval(i) for i in examples["{}_rouge1_f".format(text_column)]]
                df["art_rouges"] = art_rouges
            if data_args.reference_column is not None:
                df["summary_gen"] = examples[data_args.reference_column]
            
            df = df.progress_apply(lambda d: _tokenize_and_build_sent_index_dense(d), axis=1)
            df = df.drop(columns=["article", "summary_ext", "summary_ext_idx", "art_rouges", "summary_gen"], axis=1)
            model_inputs = {}
            for k, v in df.items():
                model_inputs[k] = v.to_list()

        elif data_args.data_preprocess == "doc_trun":
            """ Preprocess for abstractor by document truncation"""
            df = pd.DataFrame({"article": inputs})
            df = df.progress_apply(lambda d: _tokenize(d), axis=1)
            df = df.drop(columns=["article"], axis=1)
            model_inputs = {}
            for k, v in df.items():
                model_inputs[k] = v.to_list()

        elif data_args.data_preprocess == "one_doc":
            """ Preprocess for abstractor by article truncation"""
            # Add prefix
            inputs = [' '.join(inp) for inp in inputs] 

            model_inputs = tokenizer(inputs,
                                     max_length=max_source_length,
                                     padding="max_length",
                                     truncation=True)

        else:
            """ Preprocess for abstractor by article truncation"""
            # Add prefix
            #inputs = [prefix + inp for inp in inputs]
            model_inputs = tokenizer(inputs,
                                     max_length=max_source_length,
                                     padding="max_length",
                                     truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            targets = [tar if tar else " " for tar in targets]
            labels = tokenizer(targets,
                               max_length=max_target_length,
                               padding="max_length",
                               truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100
        # when we want to ignore padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100)
                                    for l in label]
                                   for label in labels["input_ids"]]

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    train_dataset = None
    eval_dataset = None
    test_dataset = None

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            # NEW: start from specified data indice
            if data_args.select_start_indice:
                start_indice = data_args.select_start_indice
                end_indice = start_indice + data_args.max_train_samples
                train_dataset = train_dataset.select(
                    range(start_indice, end_indice))
            else:
                train_dataset = train_dataset.select(
                    range(data_args.max_train_samples))
        origin_train_dataset = train_dataset
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        origin_eval_dataset = eval_dataset
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        if training_args.output_extraction_results:
            eval_dataset = {"origin": origin_eval_dataset, "processed": eval_dataset}

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(
                data_args.max_test_samples))
        origin_test_dataset = test_dataset
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        if training_args.output_extraction_results:
            test_dataset = {"origin": origin_test_dataset, "processed": test_dataset}
   
    return train_dataset, eval_dataset, test_dataset
