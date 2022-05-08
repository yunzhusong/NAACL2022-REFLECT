"""
This file is for recording the training config
"""

# -O model is for inferencing the extractor outputs (We use base/large model for training/testing)
# -A model is for building the 1st-summary reference


trained_abs_model_mapping={
    "wiki_cat_sum_animal_bl_own":{
        "base-O": "../results_for_rebuttal/wiki_cat_sum_animal_o_own/finetuned_abs/bart-base-O/checkpoint-37500",
        "large-A": "../results_for_rebuttal/wiki_cat_sum_animal_own/finetuned_abs/bart-large-cnn-A/checkpoint-21500",
        "large-O": "../results_for_rebuttal/wiki_cat_sum_animal_o_own/finetuned_abs/bart-large-cnn-O/"
    },
    "wiki_cat_sum_company_bl_own":{
        "base-O": "../results_for_rebuttal/wiki_cat_sum_company_bl_own/finetuned_abs/bart-base-O/checkpoint-47250",
        "large-A": "../results_for_rebuttal/wiki_cat_sum_company_own/finetuned_abs/bart-large-A/checkpoint-14250",
        "large-O": "../results_for_rebuttal/wiki_cat_sum_company_bl_own/finetuned_abs/bart-large-cnn-O/",
    },
    "wiki_cat_sum_film_bl_own":{
        "base-O": "../results_for_rebuttal/wiki_cat_sum_film_o_own/finetuned_abs/bart-base-O/checkpoint-40750",
        "large-A": "../results_for_rebuttal/wiki_cat_sum_film_own/finetuned_abs/bart-large-cnn-A/checkpoint-12000",
        "large-O": "../results_for_rebuttal/wiki_cat_sum_film_o_own/finetuned_abs/bart-large-cnn-O/",
    },
    "arxiv_pag_own": {
        "base-O": "../results_for_rebuttal/arxiv_own/finetuned_abs/bart-base-O/checkpoint-65250",
        "large-A": "google/pegasus-arxiv",
        "large-O": "",
    },

}
trained_ext_model_mapping={
    "multi_news_bl_own":{
        "roberta-base-squad2": "../results_for_reproduce/pretrain/roberta-base-squad2",
    },

    "wiki_cat_sum_animal_bl_own":{
        "roberta-base-squad2": "../results_for_reproduce/pretrain/roberta-base-squad2",
    },
    "wiki_cat_sum_company_bl_own":{
        "roberta-base-squad2": "../results_for_reproduce/pretrain/roberta-base-squad2",
    },
    "wiki_cat_sum_film_bl_own":{
        "roberta-base-squad2": "../results_for_reproduce/pretrain/roberta-base-squad2",
    },
    "arxiv_pag_own":{
        "roberta-base-squad2": "../results_for_reproduce/pretrain/roberta-base-squad2",
    },

}
trained_model_mapping={
    "wiki_cat_sum_animal_bl_own":{
        "mle_pretrained_rouge": "../results_for_rebuttal/wiki_cat_sum_animal_bl_own/mle_extractor/3hier/SR_POR/checkpoint-17900",
        "mle_pretrained_rouge_POR": "../results_for_rebuttal/wiki_cat_sum_animal_bl_own/mle_extractor/3hier/POR/checkpoint-16400",
        "rl_trained_rouge": "../results_for_rebuttal/wiki_cat_sum_animal_bl_own/rl_extractor/3_hier/mle_SR_POR_+_cacs_SR_full/checkpoint-1600",
        "large-O": "../results_for_rebuttal/wiki_cat_sum_animal_o_own/finetuned_abs/bart-large-cnn-O/checkpoint-30500",
        "large-A": "../results_for_rebuttal/wiki_cat_sum_animal_own/finetuned_abs/bart-large-cnn-A/checkpoint-21500",
    },
    "wiki_cat_sum_company_bl_own":{
        "mle_pretrained_rouge": "../results_for_rebuttal/wiki_cat_sum_company_bl_own/mle_extractor/3hier/SR_POR/checkpoint-7300",
        "mle_pretrained_rouge_POR": "../results_for_rebuttal/wiki_cat_sum_company_bl_own/mle_extractor/3hier/POR/checkpoint-13600",
        "rl_trained_rouge": "../results_for_rebuttal/wiki_cat_sum_company_bl_own/rl_extractor/3_hier/mle_SR_POR_+_cacs_SR_full/checkpoint-6200",
        "large-O": "../results_for_rebuttal/wiki_cat_sum_company_o_own/finetuned_abs/bart-large-cnn-O/checkpoint-23500",
        "large-A": "../results_for_rebuttal/wiki_cat_sum_company_own/finetuned_abs/bart-large-A/checkpoint-14250",
    },
    "wiki_cat_sum_film_bl_own":{
        "mle_pretrained_rouge": "../results_for_rebuttal/wiki_cat_sum_film_bl_own/mle_extractor/3hier/SR_POR/checkpoint-12500",
        "mle_pretrained_rouge_POR": "../results_for_rebuttal/wiki_cat_sum_film_bl_own/mle_extractor/3hier/POR/checkpoint-9200",
        "rl_trained_rouge": "../results_for_rebuttal/wiki_cat_sum_film_bl_own/rl_extractor/3_hier/mle_SR_POR_+_cacs_SR_full/checkpoint-10000",
        "large-O": "../results_for_rebuttal/wiki_cat_sum_film_o_own/finetuned_abs/bart-large-cnn-O/checkpoint-28500",
        "large-A": "../results_for_rebuttal/wiki_cat_sum_film_own/finetuned_abs/bart-large-cnn-A/checkpoint-12000",
    },
    "arxiv_pag_own":{
        "mle_pretrained_rouge_POR": "../results_for_rebuttal/arxiv_pag_own/mle_extractor/3hier/POR/checkpoint-5500",
        "rl_trained_rouge": "../results_for_rebuttal/arxiv_pag_own/rl_extractor/3_hier/mle_SR_POR_+_cacs_SR/checkpoint-9000_save",
        "base-O": "../results_for_rebuttal/arxiv_own/finetuned_abs/bart-base-O/checkpoint-65250",
        "large-O": "../results_for_rebuttal/arxiv_pag_own/finetuned_abs/bart-large-cnn-O/checkpoint-1250",
    },
}
