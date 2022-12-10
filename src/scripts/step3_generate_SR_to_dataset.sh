export CUDA_VISIBLE_DEVICES=0,1

dataset_name=multi_news
model_name_or_path=fackbook/bart-large
checkpoint_of_finetuned_abs=../outputs/multi_news/finetuned_abs/bart-large-A
output_gen_dir=../datasets/prepare_for_SR/$dataset_name/bart-large-A

# Inference test data
python -m torch.distributed.launch --nproc_per_node 2 main.py\
  --do_predict \
  --task_type seq2seq \
  --training_type mle \
  --dataset_name $dataset_name \
  --model_name_or_path $model_name_or_path \
  --output_dir $output_gen_dir/test_inference \
  --test_file ../datasets/ext_oracle/$dataset/test.csv\
  --predict_with_generate \
  --overwrite_output_dir \
  --output_abstraction_results \
  --data_preprocess doc_trun \
  --per_device_eval_batch_size 16 \
  --load_trained_model_from $checkpoint_of_finetuned_abs\

# Inference validation data
python -m torch.distributed.launch --nproc_per_node 2 main.py\
  --do_predict \
  --task_type seq2seq \
  --training_type mle \
  --dataset_name $dataset_name \
  --model_name_or_path $model_name_or_path \
  --output_dir $output_gen_dir/test_inference \
  --test_file ../datasets/ext_oracle/$dataset/validation.csv\
  --predict_with_generate \
  --overwrite_output_dir \
  --output_abstraction_results \
  --data_preprocess doc_trun \
  --per_device_eval_batch_size 16 \
  --load_trained_model_from $checkpoint_of_finetuned_abs\

# Inference train data
python -m torch.distributed.launch --nproc_per_node 2 main.py\
  --do_predict \
  --task_type seq2seq \
  --training_type mle \
  --dataset_name $dataset_name \
  --model_name_or_path $model_name_or_path \
  --output_dir $output_gen_dir/test_inference \
  --test_file ../datasets/ext_oracle/$dataset/train.csv\
  --predict_with_generate \
  --overwrite_output_dir \
  --output_abstraction_results \
  --data_preprocess doc_trun \
  --per_device_eval_batch_size 16 \
  --load_trained_model_from $checkpoint_of_finetuned_abs\


data_dir=../datasets/ext_oracle/multi_news
python ./data_preprocess/build_summary_referencing_dataset.py\
  --merged_data_dir $data_dir\
  --data_dir $data_dir\
  --gen_dir $output_gen_dir

