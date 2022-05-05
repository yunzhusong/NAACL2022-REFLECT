
# Input
path_to_generated_summary_train_file="PATH_TO_GENERATED_SUMMARY_FOR_TRAIN"
path_to_generated_summary_val_file="PATH_TO_GENERATED_SUMMARY_FOR_VALIDATION"
path_to_generated_summary_test_file="PATH_TO_GENERATED_SUMMARY_FOR_TEST"
data_dir="DATASET_AFTER_GETTING_ORACLE" # e.g., "../datasets/ext_oracle/multi_news_own"

# Output
merged_data_dir="DATASET_AFTER_COMBINING_GENERATED_SUMMARY"  # e.g., "../dataset/ext_oracle/multi_news_bl_own"


python ./data_preprocess/build_summary_referencing_dataset.py\
	--merged_data_dir $merged_data_dir\
	--data_dir $data_dir\
	--gen_train_file $path_to_generated_summary_train_file\
	--gen_validation_file $path_to_generated_summary_for_val_file\
	--gen_test_file $path_to_generated_summary_for_test_file\

# 
#python ./data_preprocess/build_summary_referencing_dataset.py\
#	--merged_data_dir ../dataset/ext_oracle/xscience_bl_own\
#	--data_dir ../datasets/ext_oracle/xscience_own\
#	--gen_train_file ../outputs/xscience_own/bart-large-A/checkpoint-1000/train_inference/test_generations.csv\
#	--gen_validation_file ../outputs/xscience_own/bart-large-A/checkpoint-1000/validation_inference/test_generations.csv\
#	--gen_test_file ../outputs/xscience_own/bart-large-A/checkpoint-1000/test_inference/test_generations.csv\
