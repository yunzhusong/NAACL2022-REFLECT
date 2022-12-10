""" Arguments for running. """
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import Seq2SeqTrainingArguments


@dataclass
class CustomSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    """
    Customized sequence-to-sequence training arguments.
    """
    multi_documents: Optional[bool] = field(
        default=True,
        metadata={
            "help":
                "Whether each datum includes multiple documents" 
        },
    )
    debug_mode: Optional[bool] = field(
        default=False,
        metadata={
            "help":
                "debug" 
        },
    )
    # Overwrite TrainingArguments default value
    remove_unused_columns: bool = field(
        default=False,
        metadata={
            "help":
                "Remove columns not required by the model when using an nlp.Dataset."
        },
    )
    # Set generation max length for evaluation during training
    train_val_max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help":
                "This argument is used to override the ``max_length`` param of ``model.generate``, which is used "
                "in the evaluation() function calls during _maybe_log_save_evaluate()."
        },
    )
    save_model_accord_to_rouge: bool = field(
        default=False,
        metadata={
            "help":
                "Whether to save model according to ROUGE-1 score instead of loss."
        },
    )
    # NEW: for loading different model
    training_type: str = field(
        default="mle",
        metadata={
            "help":
                "Use the [mle/ext_mle/ext_rl/rl/retri_mle] to train the model. Used in pipeline/build_trainer.py"
        },
    )
    # NEW: for loading different trainer
    task_type: str = field(
        default=None,
        metadata={
            "help":
                "Specify the task type in [seq2seq, two_stage_extraction] Used in main.py"
        },
    )
    # NEW: for ext-abs model setting
    different_base_model_for_two_stage: bool = field(
        default=False,
        metadata={
            "help":
                "When the model types of extractor and abstractor are different, set to `True`.\
                Meanwhile, make sure ext_model_name_or_path is also assigned. Used in main.py"
        },
    )
    train_only: str = field(
        default='',
        metadata={
            "help":
                "Specify the trainable layer"
        },
    )
    # NEW: for reflect summarization
    reference_extraction: Optional[bool] = field(
        default=False,
        metadata={
            "help":
                "Whether to extract the sentences according to the previous generated summary. Used in model/refect.py Default=False"
        },
    )
    score_regression: Optional[bool] = field(
        default=False,
        metadata={
            "help":
                "Whether to learn the rouge score as the auxilliary task. Used in pipeline/trainer_ext_*"
        },
    )
    score_cls_weighting: Optional[bool] = field(
        default=False,
        metadata={
            "help":
                "Whether to use the weighting cross entropy. Used in pipeline/trainer_ext_mle.py data/build_datasets.py"
        },
    )
    data_level: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "Data-level: article, section or None. Used in model/reflect.py Default=None"
        },
    )
    num_hierarchical_layer: Optional[int] = field(
        default=3,
        metadata={
            "help":
                "Number of hierarchical layers in extractor, 0 means flat structure for controlling loading pretrained model. Used in main.py Default=3"
        },
    )
    # NEW: for refect rl training
    update_full_action: Optional[bool] = field(
        default=True,
        metadata={
            "help":
                "Whether to update the full actions. Used in pipeline/trainer_ext_rl.py Default=True"
        },
    )
    use_ext_reward: bool = field(
        default=False,
        metadata={
            "help":
                "Whether to consider the extraction results as one of the reward"
        },
    )
    use_mixer_loss: bool = field(
        default=True,
        metadata={
            "help":
                "Add MLE loss and RL loss to create mixer loss."
        },
    )
    mixer_weight: float = field(
        default=0.1,
        metadata={
            "help":
                "Loss weight for RL loss in mixer loss. "
        },
    )
    # NEW: for reflect rl testing
    predict_with_generate_abs: Optional[bool] = field(
        default=True,
        metadata={
            "help":
                "Whether to predict the abstractor generation results during prediction step. Used in pipeline/trainer_ext_rl.py Set to False during inference can accelerate the inference speed. Default=True."
        },
    )
    output_extraction_results: Optional[bool] = field(
        default=False,
        metadata={
            "help":
                "Whether to write out the extraction results. Used in pipeline/trainer_ext_rl.py and process.py Default=False"
        },
    )
    output_abstraction_results: Optional[bool] = field(
        default=False,
        metadata={
            "help":
                "Whether to write out the generation results. Used in process.py Default=False"
        },
    )
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help":
                "Path to pretrained model or model identifier from huggingface.co/models"
        })
    # NEW: If model types of extractor and abstractor are differnt, use this argument to specify extractor.
    ext_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "Path to pretrained model or model identifier from huggingface.co/models.\
                This is to specify the extractor model."
        })
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "Pretrained config name or path if not the same as model_name"
        })
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "Pretrained tokenizer name or path if not the same as model_name"
        })
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help":
                "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help":
                "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help":
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
        },
    )
    # NEW: load_trained_model
    load_trained_model_from: Optional[str] = field(
        default=False,
        metadata={
            "help":
                "Specified a pre-trained model path, this argument only load the model checkpoint."
        },
    )

    load_trained_extractor_from: Optional[str] = field(
        default=False,
        metadata={
            "help":
                "Specified a pre-trained model path, this argument only load the model checkpoint."
        },
    )

    load_trained_abstractor_from: Optional[str] = field(
        default=False,
        metadata={
            "help":
                "Specified a pre-trained model path, this argument only load the model checkpoint."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library)."
        })
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "The configuration name of the dataset to use (via the datasets library)."
        })
    text_column: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "The name of the column in the datasets containing the full texts (for summarization)."
        },
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "The name of the column in the datasets containing the summaries (for summarization)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "The input training data file (a jsonlines or csv file)."
        },
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "An optional input evaluation data file to evaluate the metrics (rouge) on "
                "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "An optional input test data file to evaluate the metrics (rouge) on "
                "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=True,
        metadata={"help":
                  "Overwrite the cached training and evaluation sets"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=1,
        metadata={
            "help": "The number of processes to use for the preprocessing."
        },
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help":
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help":
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help":
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help":
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help":
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help":
                "For debugging purposes or quicker training, truncate the number of validation examples to this "
                "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help":
                "For debugging purposes or quicker training, truncate the number of test examples to this "
                "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help":
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help":
                "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "A prefix to add before every source text (useful for T5 models)."
        },
    )

    # NEW
    eval_dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "The name of the validation dataset to use (via the datasets library)."
        },
    )
    shuffle_before_select: bool = field(
        default=True,
        metadata={
            "help":
                "Whether to shuffle the dataset before select data. This argument works for all dataset splits."
        },
    )
    select_start_indice: Optional[int] = field(
        default=0,
        metadata={
            "help":
                "The first data indice for selection. This argument only works for training set."
        },
    )
    # NEW: for deciding different data preprocessing
    data_preprocess: str = field(
        default='',
        metadata={
            "help":
                "How to preprocess the input documents, assign 'doc_trun' to do document-wise truncation, assign 'doc_trun_and_build_sent_index' for further adding the sentence index; assign '' for the plain process. Default=''"
        },
    )
    # NEW: for ext-abs framework and reflect model
    summary_ext_column: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "The name of the column in the datasets containing the summaries (for summarization)."
        },
    )
    reference_column: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "The name of the column in the datasets containing the reference extraction (for summarization)."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None and self.test_file is None:
            raise ValueError(
                "Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv", "json"
                ], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv", "json"
                ], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in [
                    "csv", "json"
                ], "`test_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length
