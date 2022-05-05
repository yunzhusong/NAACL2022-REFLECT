export CUDA_VISIBLE_DEVICES=0

dataset="DATASETNAME" # e.g., multi_news_own
checkpoint_to_finetuned_abs="ASSIGN_THE_MODEL_CHECKPOINT"

python main.py\
  --task_type seq2seq\
  --training_type mle\
  --dataset_name $dataset\
  --model_name_or_path facebook/bart-large\
  --load_trained_model_from $checkpoint_to_finetuned_abs\
  --output_dir $checkpoint_to_finetuned_abs/test_inference\
  --test_file ../datasets/ext_oracle/$dataset/test.csv\
  --predict_with_generate\
  --overwrite_output_dir\
  --output_abstraction_results\

python main.py\
  --task_type seq2seq\
  --training_type mle\
  --dataset_name $dataset\
  --model_name_or_path facebook/bart-large\
  --load_trained_model_from $checkpoint_to_finetuned_abs\
  --output_dir $checkpoint_to_finetuned_abs/validation_inference\
  --test_file ../datasets/ext_oracle/$dataset/validation.csv\
  --predict_with_generate\
  --overwrite_output_dir\
  --output_abstraction_results\

python main.py\
  --task_type seq2seq\
  --training_type mle\
  --dataset_name $dataset\
  --model_name_or_path facebook/bart-large\
  --load_trained_model_from $checkpoint_to_finetuned_abs\
  --output_dir $checkpoint_to_finetuned_abs/train_inference\
  --test_file ../datasets/ext_oracle/$dataset/train.csv\
  --predict_with_generate\
  --overwrite_output_dir\
  --output_abstraction_results\
