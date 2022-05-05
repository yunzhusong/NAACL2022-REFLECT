# NAACL2022-REFLECT

Code for the paper: Improving Multi-Document Summarization through Referenced Flexible Extraction with Credit-Awareness
![MDS_framework](https://user-images.githubusercontent.com/45812808/164428295-66af2bfd-3e07-4e2d-a3c8-ecdd56df7857.png)

Author: [@Yun-Zhu Song](http://github.com/yunzhusong), [@Yi-Syuan Chen](https://github.com/YiSyuanChen),Hong-Han Shuai

The preprocessed datasets and pretrained model will be released soon.

---
## Environment Requirements for Your Reference
```
pip install -r requirements.txt
```

## Dataset Preparation


1. Multi-News
```
python ./data_download/output_dataset.py\
  --output_dir ../datasets/origin/multi_news\
  --dataset_name multi_news\
```

2. Milti-XScience
```
python ./data_download/output_dataset.py\
  --output_dir ../datasets/origin/xscence\
  --dataset_name multi_x_science_sum\
```
3. WikiCatSum (NOTE: The version of transformers is 4.12.5)
```
python ./data_download/output_dataset.py\
  --output_dir ../datasets/origin/wikicatsum/animal\
  --dataset_name GEM/wiki_cat_sum\
  --dataset_config animal\
  
python ./data_download/output_dataset.py\
  --output_dir ../datasets/origin/wikicatsum/company\
  --dataset_name GEM/wiki_cat_sum\
  --dataset_config company\
  
python ./data_download/output_dataset.py\
  --output_dir ../datasets/origin/wikicatsum/film\
  --dataset_name GEM/wiki_cat_sum\
  --dataset_config film\
```
## Trained Model

|            | Finetuned Abstractor | Pretrained | Final |
|------------|----------------------|------------|-------|
| Multi-News | [Bart-Base-Oracle](https://drive.google.com/file/d/1ELzt-EjzmXhK0vKiAOy-nEkV_VxUGkYi/view?usp=sharing), [Bart-Largs-Oracle](https://drive.google.com/file/d/13UPz6_AdVpxrjj-uJxhbKLL0ZUGjdTGx/view?usp=sharing) | [download](https://drive.google.com/file/d/1-tNFQs6BNKlCJl4LGJ8SGpjHH1an5kfR/view?usp=sharing) | [download](https://drive.google.com/file/d/14lp4ViPDJlYZScQc5R4N7Y5Oje1-YShi/view?usp=sharing)|


## Training

### 1. Abstractor Training

There are 4 different configs for abstractor.

| Model Size | Input Type |
|------------|------------|
| BART Base  | Oracle     |
| BAET Base  | Article    | 
| BART Large | Oracle     |
| BAET Large | Article    |


How to change to different configs

| dataset_name          | Oracle Text Column | Article Text Column |
|-----------------------|--------------------|---------------------|
| multi_news_bl_art_own | summary_ext        | document            | 
| xscience2_bl_own      | summary_ext        | document            |

```
python main.py ./scirpts/args/finetine_abs.json
```

### 2. Extractor Pretraining

```
python main.py ./scripts/args/train_ext_mle.json
```

### 3. Extractor Training

```
python main.py ./scripts/args/train_ext_rl.json
```

### 4. Model Evaluation
```
python main.py ./scripts/args/pred.json
python main.py ./scripts/args/eval.json
```

## Argument Description

**Arguments for switching between abstractor training or extractor training**
```
"task_type": "seq2seq" for abstractor. "two_stage_extraction" for extractor.
"training_type": "mle" for abstractor finetuning. "ext_mle" for extractor pretraining. "ext_rl" for extractor training.
"data_preprocess": "doc_trun" for abstractor. "doc_trun_and_build_sent_index" for extractor.
```
**Arguments for extractor only**
```
"summary_ext_column": "summary_ext"
```
**Arguments for training our extractor:**
```
"ext_model_name_or_path": Specify the model name or path to give the extractor config. default: deepset/roberta-base-squad2.
"different_base_model_for_two_stage": Specify true when the extractor config and abstractor config are different. default: true.
"load_trained_abstractor_from": Specify the model path for finetuned abstractor.
"load_trained_extractor_from": Specify the model path for pretrained extractor.
"train_only": Specify module name for training the module. default:"extractor"
```
**Arguments for model configuration:**
```
"score_cls_weighting": whether to adopt Peudo Oracle Relaxation (POR), true or false.
"reference_extraction": wether to adopt Summary Referencing (SR), true or false. If true, need to assign the "reference_column" to the column of pregenearted summary.
"reference_column": Assign the column of pregenerated summary in dataset. Only activate when \"reference_extraction\" is true. default: "summary_gen".
""num_hierarchical_layer"": Number of hierarchical layers in extractor, 0 means flat structure for controlling loading pretrained model. Used in main.py. default:3.
```
**Arguments for reinforcement learning:**
```
"use_mixer_loss": Whether to consider the MLE loss. dedault: true.
"mixer_weight": The weight for mixing the MLE and RL loss. default: 0.1.
"update_full_action": Wether to update the full action or only update the output with the sampled action that are different from the greedy action. false for CASC, true for SC.
```
