export CUDA_VISIBLE_DEVICES=7

#### 0. Finetune bart-large with article input for preparing summary reference
# or use public source, such as `nikhedward/bart-large-cnn-finetuned-multi-news` 
#python main.py ./scripts/args/finetune_abs_large-A.json

#### 1. Finetune abstractor with oracle input
#python main.py ./scripts/args/finetune_abs_base-O.json
#python main.py ./scripts/args/finetune_abs_large-O.json

#### 2. Pretrain extractor
# assign the checkpoint of finetune_abs_base-O
#python main.py ./scripts/args/train_ext_mle.json

#### 3. Train extractor
#python main.py ./scripts/args/train_ext_rl.json

#### 4. Obtain the predicted extraction results
#python main.py ./scripts/args/pred.json

#### 5. Evaluate the extraction results
python main.py ./scripts/args/eval.json
