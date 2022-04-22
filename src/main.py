""" Fine-tunes the Huggingface models for sequence to sequence.

The code is developed based on :
    https://github.com/huggingface/transformers/tree/v4.4.2/examples/seq2seq
"""

import logging
import os
import sys
import copy

import torch
torch.autograd.set_detect_anomaly(True) 
import transformers
from transformers import (
    CONFIG_MAPPING, AutoConfig,AutoTokenizer, HfArgumentParser, set_seed, AutoModelForSeq2SeqLM
)
from transformers.trainer_utils import is_main_process
from transformers.utils import check_min_version

from data.build_datasets import build_datasets
from pipelines.build_trainer import build_trainer
from others.arguments import ModelArguments, DataTrainingArguments, CustomSeq2SeqTrainingArguments
from others.config import trained_abs_model_mapping, trained_ext_model_mapping, trained_model_mapping
from processes import train_process, eval_process, predict_process

# rl learning
from models.generation_utils import generate_with_grad

import shutil
import pdb

check_min_version("4.5.0")
logger = logging.getLogger(__name__)

def main():

    # Parse arguments
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomSeq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        os.makedirs(training_args.output_dir, exist_ok=True)
        json_file_name = os.path.basename(sys.argv[1])
        shutil.copyfile(sys.argv[1], "{}/{}".format(training_args.output_dir, json_file_name))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("Task type: {}".format(training_args.task_type))
    if training_args.do_train:
        print("Training type: {}".format(training_args.training_type))
        print("Summary Reference: {}".format(training_args.reference_extraction))
        if training_args.training_type=="ext_mle": 
            print("Pseudo Oracle Relaxation: {}".format(training_args.score_cls_weighting))
        elif training_args.training_type=="ext_rl":
            print("Credit-Aware Self-Critic: {}".format(not training_args.update_full_action))
        elif training_args.training_type=="mle":
            print("Train Abstractorwith {} as input".format(data_args.text_column))
        else:
            raise ValueError("--task_type sould be one of `ext_mle/ext_rl/mle`")

    # Set the verbosity to warning of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_warning()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.WARN if is_main_process(training_args.local_rank
                                                   ) else logging.WARN)
    logger.info("Training/evaluation parameters", training_args)

    # Log on each process the small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, "+
        f"device: {training_args.device}, "+
        f"n_gpu: {training_args.n_gpu}, "+
        f"distributed training: {bool(training_args.local_rank != -1)}, "+
        f"16-bits training: {training_args.fp16}") 

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # NOTE: Load config *************************************************** ####
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name,
                                            **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path,
                                            **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]
        logger.warning(
            "You are instantiating a new config instance from scratch.")
    if 'bart-large' in model_args.model_name_or_path:  # NEW: for bart-large (not sure why)
        config.forced_bos_token_id = config.bos_token_id

    # NOTE: Load tokenizer ************************************************ ####
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name,
                                                  **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                                  **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this scripts."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # NOTE: Use different models for two stage learning ******************* ####
    if training_args.different_base_model_for_two_stage:
        ext_config = AutoConfig.from_pretrained(model_args.ext_model_name_or_path,
                                               **config_kwargs)
        ext_model_kwargs = {
            "from_tf": bool(".ckpt" in model_args.ext_model_name_or_path),
            "config": ext_config,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
        }
    else:
        ext_config = config

    # NOTE: Build model for a specific task ******************************** ####
    model_kwargs = {
        "from_tf": bool(".ckpt" in model_args.model_name_or_path),
        "config": config,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    if training_args.task_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )
        model.additional_input_kwargs = []

    elif training_args.task_type == "two_stage_extraction":
        """ Load the pretrained model by model_args.load_trained_abstractor(extractor)_from """
        from models.reflect import HierarchicalRobertaForTokenClassification, ReflectModel

        abstractor = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )

        if training_args.num_hierarchical_layer > 0:

            sec_ext_num_hidden_layers = training_args.num_hierarchical_layer

            art_ext_config = copy.deepcopy(ext_config)
            art_ext_config.num_hidden_layers = ext_config.num_hidden_layers - sec_ext_num_hidden_layers

            sec_ext_config = copy.deepcopy(ext_config)
            sec_ext_config.num_hidden_layers = sec_ext_num_hidden_layers

            article_extractor = HierarchicalRobertaForTokenClassification(art_ext_config, training_args)
            section_extractor = HierarchicalRobertaForTokenClassification(sec_ext_config, training_args)

            model = ReflectModel(training_args, tokenizer, abstractor, article_extractor, section_extractor)
        else:
            extractor = RobertaForTokenClassifier(ext_config, training_args)
            model = ReflectModel(training_args, tokenizer, abstractor, extractor)

    else:
        ValueError(
            "Please specify the training_args.task_type in [seq2seq, two_stage_extraction]"
        )

    # NOTE: Load pretrained model  ************************************** ####

    if model_args.load_trained_model_from:
        ckpt_path = os.path.join(model_args.load_trained_model_from, "pytorch_model.bin")
        if not os.path.exists(ckpt_path):
            trained_models = trained_model_mapping[data_args.dataset_name]
            trained_model = trained_models[model_args.load_trained_model_from] 
            ckpt_path = os.path.join(trained_model, "pytorch_model.bin")
        ckpt = torch.load(ckpt_path)
        keys = model.load_state_dict(ckpt, strict=False)
        logger.warning("Load the specified trained model")
        print(keys)

    if model_args.load_trained_extractor_from:
        try:
            from transformers import AutoModelForQuestionAnswering
            ckpt = AutoModelForQuestionAnswering.from_pretrained(model_args.load_trained_extractor_from).state_dict()
            logger.warning("Load the pretrained model from huggingface")
        except:
            ckpt_path = os.path.join(model_args.load_trained_extractor_from, "pytorch_model.bin")
            if not os.path.exists(ckpt_path):
                trained_extractors = trained_ext_model_mapping[data_args.dataset_name]
                trained_extractor = trained_extractors[model_args.load_trained_extractor_from] 
                ckpt_path = os.path.join(trained_extractor, "pytorch_model.bin")
            ckpt = torch.load(ckpt_path)
            logger.warning("Load the trained extractor from local file")
            
        new_ckpt = {}
        for k, v in ckpt.items():
            # The ckpt contains extractor and abstractor
            extractor_split = k.split("extractor.")
            if len(extractor_split)>1:
                new_ckpt[extractor_split[-1]] = v

        if len(new_ckpt)==0:
            # The ckpt contains only extractor
            new_ckpt = ckpt

        # NOTE: subsequencial load the weights
        if training_args.num_hierarchical_layer > 0:
            keys = model.section_extractor.load_state_dict(new_ckpt, strict=False)
            logging.warning("Load Section Extractor")
            num_exist_layers = training_args.num_hierarchical_layer
            origin_ckpt = new_ckpt
            new_ckpt = {}
            for k in keys.unexpected_keys:
                # Rename the keys
                num = k.split(".")[3]
                new_k = '{}'.format(int(num)-num_exist_layers).join(k.split(num))
                new_ckpt[new_k] = origin_ckpt[k]

            for k in ['roberta.embeddings.position_ids', 
                      'roberta.embeddings.word_embeddings.weight',
                      'roberta.embeddings.position_embeddings.weight',
                      'roberta.embeddings.token_type_embeddings.weight',
                      'roberta.embeddings.LayerNorm.weight',
                      'roberta.embeddings.LayerNorm.bias',
                      'qa_outputs.weight',
                      'qa_outputs.bias']:
                new_ckpt[k] = origin_ckpt[k]

        keys = model.extractor.load_state_dict(new_ckpt, strict=False)
        logging.warning("Load Extractor:")
        print(keys)

    if model_args.load_trained_abstractor_from:
        ckpt_path = os.path.join(model_args.load_trained_abstractor_from, "pytorch_model.bin")
        if not os.path.exists(ckpt_path):
            trained_abstractors = trained_abs_model_mapping[data_args.dataset_name]
            trained_abstractor = trained_abstractors[model_args.load_trained_abstractor_from] 

            ckpt_path = os.path.join(trained_abstractor, "pytorch_model.bin")
        ckpt = torch.load(ckpt_path)
        #keys = model.abstractor.model.load_state_dict(ckpt, strict=False)
        #keys = model.abstractor.load_state_dict(ckpt, strict=False)
        logger.warning("Load the trained abstractor from local file")
        keys = model.abstractor.load_state_dict(ckpt, strict=True)
        print(keys)

    # NOTE: Check-list **************************************************** ####
    if training_args.task_type=='seq2seq' and model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    if training_args.label_smoothing_factor > 0 and not hasattr(
            model, "prepare_decoder_input_ids_from_labels"):
        logger.warn(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    #### ****************************************************************** ####

    # For rl training, add a method to an existing model
    if 'rl' in training_args.training_type:
        model.generate_with_grad = generate_with_grad

    # Build datasets
    train_dataset, eval_dataset, test_dataset = build_datasets(
        data_args, training_args, tokenizer)

    # Build trainer
    trainer = build_trainer(model_args, data_args, training_args, model,
                            tokenizer, train_dataset, eval_dataset, test_dataset)
    test_dataset = test_dataset["processed"] if training_args.output_extraction_results else test_dataset

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        train_process(model_args, data_args, training_args, trainer,
                      train_dataset)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluation ***")
        eval_process(data_args, trainer, eval_dataset)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Prediction ***")
        predict_process(data_args, training_args, trainer, test_dataset,
                        tokenizer)


if __name__ == "__main__":
    main()
