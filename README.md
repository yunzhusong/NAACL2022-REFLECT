# NAACL2022-REFLECT

Code for the paper: [Improving Multi-Document Summarization through Referenced Flexible Extraction with Credit-Awareness](https://aclanthology.org/2022.naacl-main.120.pdf)
![MDS_framework](https://user-images.githubusercontent.com/45812808/164428295-66af2bfd-3e07-4e2d-a3c8-ecdd56df7857.png)

Author: [@Yun-Zhu Song](http://github.com/yunzhusong), [@Yi-Syuan Chen](https://github.com/YiSyuanChen),Hong-Han Shuai

The preprocessed datasets and pretrained model will be released soon.

---
## Referenced Environment Setup
```
pip install -r requirements.txt
```

## Dataset Preparation
### Option 1.
NOTE:
1. We provide an example for processing multi-news end-to-end **src/scripts/construct_dataset_end2end.sh**
2. The names of datasets can be found in **src/data/build_datasets.py**.


Steps: (1) download the dataset; (2) get the pseudo extractio oracle and rouge scores for each document sentence; (3) generate summary from the fine-tuned abstractor and merge the generated summary to the dataset. 

(1) download dataset
```
cd src
./scripts/step1_download_dataset.sh
```

(2) get pseudo extraction (take multi_news as examples)
```
cd src
./scripts/step2_build_POR_label.sh
```

(3) generate summary from finetuned abstractor ([multi_news](https://drive.google.com/file/d/1EDl-HZLQDPWTy9ZMxWvxHPNI_TTnKlYm/view?usp=sharing)) and merge the generated results to dataset. (take multi_news as examples)
```
cd src
./scripts/step3_generate_SR_to_dataset.sh
```


### Option 2. Dowload Our Processed Dataset
Please place the dataset at **datasets/ext_oracle/** according to the following code structure or change the dataset directory path writing in **src/data/build_datasets.py**.
<!--
Please sent an email to Yun-Zhu Song (yzsong.ee07@nycu.edu.tw) to request our processed dataset.
-->

[Multi-News](https://drive.google.com/file/d/1i8JuegEmGik-MhEU9GsKy3KcaJSr_k-I/view?usp=sharing),
[Xscience](https://drive.google.com/file/d/1R5eyDaCtorCh14yijqfCyduCjffYv8Ne/view?usp=sharing),
[WikiCatSum](https://drive.google.com/file/d/1Q6IVCf2nUFLtlW1oX4l5B6_tcWLUuLHA/view?usp=sharing)

#### Code Structure
```
src\
  |_main.py -> main function
  |_process.py -> for defining different operation process
  |_scripts
    |_args\
      |_finetune_abs_base_O.json -> configuration of finetuning abstractor (base, oracle input) for supporting extractor RL training
      |_finetune_abs_large_O.json -> configuration of finetuning abstractor (large, oracle input) for test time inference
      |_finetune_abs_large_A.json -> configuration of finetuning abstractor (large, article input) for providing summary reference
      |_train_ext_mle.json -> configuration of training extractor with MLE (extractor pretraining)
      |_train_ext_rl.json -> configuration of training extractor with RL (extractor training)
      |_pred.json -> configuration of obtaining the extraction prediction
      |_eval.json -> configuration of evaluating the extraction results
    |_run.sh -> recording the scripts for the training and evaluation steps
    |_construct_dataset_end2end.sh -> an example for constructing the multi_news end-to-end

datasets\
  |_origin\
    |_multi_news\
    |_xscience\
    |_wikicatsum\
  |_ext_oracle\ -> put the processed datasets in this directory
    |_multi_news\
    |_xscience\
    |_wikicatsum\
      |_animal\
      |_company\
      |_film\
      
outputs\ -> directory for saving experiments
  |_multi_news\
    |_finetuned_abs\
      |_bart-base-O\ -> for supporting extractor RL training
      |_bart-large-O\ -> for test time inference
      |_bart-large-A\ -> for generating summary reference
    |_extractor_mle\
      |_SR_POR\ -> pretrained extractor
    |_extractor_rl\
      |_final\ -> final model
    
```
Download the datasets and put



## Trained Model

|   Dataset  | Finetuned Abstractor | Pretrained (REFLECT-MLE) | Final (REFLECT) |
|------------|----------------------|--------------------------|-----------------|
| Multi-News | [Bart-base-Oracle](https://drive.google.com/file/d/12RlJUo0Yp8J9tkgJBpGyBcoPBevif1JL/view?usp=sharing), [Bart-large-Oracle](https://drive.google.com/file/d/1VONOaQQhWe0RG2ogGlRsnUCQSe29ac9o/view?usp=sharing), [Bart-large-Article](https://drive.google.com/file/d/1EDl-HZLQDPWTy9ZMxWvxHPNI_TTnKlYm/view?usp=sharing) | [SR_POR](https://drive.google.com/file/d/1bI0tiJN3fqI22eTRWykiv4yEWTq9oBaM/view?usp=sharing) | [final](https://drive.google.com/file/d/1tZqtDb7wzZgTxJVWZsalsrCxratxNKgV/view?usp=sharing) |



<!--
|------------|----------------------|--------------------------|-----------------|
| Multi-News | [Bart-Base-Oracle](https://drive.google.com/file/d/1MEouMEzWtzJ9du4w6-wCkmJcDg8jOHzw/view?usp=sharing), [Bart-Large-Oracle](https://drive.google.com/file/d/1VONOaQQhWe0RG2ogGlRsnUCQSe29ac9o/view?usp=sharing) | [download](https://drive.google.com/file/d/1-0YqMCdwwzkS4IafL0aM5QJqQLiWIUml/view?usp=sharing) | [download](https://drive.google.com/file/d/1tZqtDb7wzZgTxJVWZsalsrCxratxNKgV/view?usp=sharing)|

| WikiCatSum/animal | [Bart-Base-Oracle]()[Bart-Large-Oracle]|[download]()|[download]()|

## Predictions
| Dataset | BART-Large | REFLECT |
|---------|------------|---------|
| WikiCatSum |[Animal](https://drive.google.com/file/d/1PP5nXdXSjH4jy6J0fjlDVZXJ-h9LoWf5/view?usp=sharing), [Company](https://drive.google.com/file/d/1nUgWnnzsGQvAqR8cj8hYqPL37CaeogpJ/view?usp=sharing), [Film](https://drive.google.com/file/d/1xxTevhR2pqcbh9mjuJG2GUHsf63OZDzf/view?usp=sharing)|[Animal](https://drive.google.com/file/d/1mlFr_5ukU7e3AIEPHDfhv3PBVpckp92U/view?usp=sharing), [Company](https://drive.google.com/file/d/1gQI541wJfIA260ZgO-b7JIORGbofIG2E/view?usp=sharing), [Film](https://drive.google.com/file/d/1KHdyBz7TjE4BwUAkq58BdMXJvlmOEs6c/view?usp=sharing) |

[WikiCatSum](https://drive.google.com/drive/folders/1CSt5VORNeB1-fAqk4GAts0Jp9VYyfImP?usp=sharing)
-->

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
| multi_news_bl_own     | summary_ext        | document            | 
| xscience_bl_own       | summary_ext        | document            |

```
cd src
python main.py ./scirpts/args/finetune_abs_base_O.json
python main.py ./scirpts/args/finetine_abs_large_O.json
python main.py ./scirpts/args/finetine_abs_large_A.json
```

### 2. Extractor Pretraining

```
cd src
python main.py ./scripts/args/train_ext_mle.json
```

### 3. Extractor Training

```
cd src
python main.py ./scripts/args/train_ext_rl.json
```

### 4. Model Evaluation
```
cd src
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

---
Citation
```
@inproceedings{song-etal-2022-improving,
    title = "Improving Multi-Document Summarization through Referenced Flexible Extraction with Credit-Awareness",
    author = "Song, Yun-Zhu  and
      Chen, Yi-Syuan  and
      Shuai, Hong-Han",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.120",
    doi = "10.18653/v1/2022.naacl-main.120",
    pages = "1667--1681",
}
```
