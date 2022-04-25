export CUDA_VISIBLE_DEVICES=1

# 1. Finetune abstractor with oracle input
#python main.py ./scripts/args/finetune_abs.json

# 2. Pretrain extractor
#python main.py ./scripts/args/train_ext_mle.json

# 3. Train extractor
#python main.py ./scripts/args/train_ext_rl.json

# 4. Obtain the predicted extraction results
python main.py ./scripts/args/pred.json

# 5. Evaluate the extraction results
python main.py ./scripts/args/eval.json
