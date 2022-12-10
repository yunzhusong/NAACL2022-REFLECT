export CUDA_VISIBLE_DEVICES=0,1

dataset_name=multi_news 
raw_data_dir=../datasets/origin/$dataset_name
aft_data_dir=../datasets/ext_oracle/$dataset_name

#### 1. Download Raw Dataset
#python ./data_download/output_dataset.py\
#  --dataset_name $dataset_name\
#  --output_dir $raw_data_dir\

#### 2. Build Pseudo Extraction Label
# It takes a while to perform greedy selection
#python ./data_preprocess/build_pseudo_extraction_oracle.py \
#  --dataset $dataset_name \
#  --test_file $raw_data_dir/test.csv \
#  --train_file $raw_data_dir/train.csv \
#  --validation_file $raw_data_dir/validation.csv \
#  --output_dir $aft_data_dir \
#  --sent_num 20 \

#### 3. Build Summary Reference
# Get the summary reference from pretrained model (can repalce to any other information)
# It takes some time to inference model

model_name_or_path=nikhedward/bart-large-cnn-finetuned-multi-news
output_gen_dir=../datasets/prepare_for_SR/$dataset_name/$model_name_or_path

#python -m torch.distributed.launch --nproc_per_node 2 main.py\
#  --do_predict \
#  --task_type seq2seq \
#  --training_type mle \
#  --dataset_name $dataset_name \
#  --model_name_or_path $model_name_or_path \
#  --output_dir $output_gen_dir/test_inference \
#  --test_file $aft_data_dir/test.csv \
#  --predict_with_generate \
#  --overwrite_output_dir \
#  --output_abstraction_results \
#  --data_preprocess doc_trun \
#  --per_device_eval_batch_size 16 \
#  #--load_trained_model_from $checkpoint_of_finetune_large_A\

#python -m torch.distributed.launch --nproc_per_node 2 main.py\
#  --task_type seq2seq\
#  --training_type mle\
#  --dataset_name $dataset_name\
#  --model_name_or_path $model_name_or_path\
#  --output_dir $output_gen_dir/validation_inference \
#  --test_file $aft_data_dir/validation.csv\
#  --predict_with_generate\
#  --overwrite_output_dir\
#  --output_abstraction_results\
#  #--load_trained_model_from $checkpoint_of_finetune_large_A\

#python -m torch.distributed.launch --nproc_per_node 2 main.py\
#  --task_type seq2seq\
#  --training_type mle\
#  --dataset_name $dataset_name\
#  --model_name_or_path $model_name_or_path\
#  --output_dir $output_gen_dir/train_inference \
#  --test_file $aft_data_dir/train.csv\
#  --predict_with_generate\
#  --overwrite_output_dir\
#  --output_abstraction_results\
#  #--load_trained_model_from $checkpoint_of_finetune_large_A\

#### 4. Combining the generated summaries (summary reference) to dataset
#python ./data_preprocess/build_summary_referencing_dataset.py\
#  --merged_data_dir $aft_data_dir\
#  --data_dir $aft_data_dir\
#  --gen_dir $output_gen_dir


