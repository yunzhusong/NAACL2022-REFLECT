export CUDA_VISIBLE_DEVICES=1

# NOTE: should first prepare csv files from OutputDataset

#python ./data_preprocess/extractive_oracle_by_generation_xscience.py \
#	--dataset multi_news \
#	--output_dir ../datasets/ext_oracle/with_doc_split_sep/xscience2_own \
#	--sent_num 20 \
#	--max_sent_len 1024 \
#
########## Extractive Oracle  ##########
#python ./data_preprocess/extractive_oracle_by_generation.py \
#	--dataset scientific_papers \
#	--train_file ../datasets/origin/arxiv_own/train.csv \
#	--validation_file ../datasets/origin/arxiv_own/validation.csv \
#	--test_file ../datasets/origin/arxiv_own/test.csv \
#	--output_dir ../datasets/ext_oracle/arxiv_own \
#	--sent_num 30 \
#	--max_sent_len 1024 \
#
#python ./data_preprocess/extractive_oracle_by_generation.py \
#	--dataset wiki_cat_sum \
#	--train_file ../datasets/origin/wikicatsum/animal/train.csv \
#	--validation_file ../datasets/origin/wikicatsum/animal/validation.csv \
#	--test_file ../datasets/origin/wikicatsum/animal/test.csv \
#	--output_dir ../datasets/ext_oracle/wikicatsum/animal_own \
#	--sent_num 30 \
#	--max_sent_len 1024 \
#
python ./data_preprocess/extractive_oracle_by_generation.py \
	--dataset wiki_cat_sum \
	--train_file ../datasets/origin/wikicatsum/company/train.pkl \
	--validation_file ../datasets/origin/wikicatsum/company/validation.pkl \
	--test_file ../datasets/origin/wikicatsum/company/test.pkl \
	--output_dir ../datasets/ext_oracle/wikicatsum/company_own \
	--sent_num 30 \
	--max_sent_len 1024 \

#python ./data_preprocess/extractive_oracle_by_generation.py \
#	--dataset wiki_cat_sum \
#	--train_file ../datasets/origin/wikicatsum/film/train.csv \
#	--validation_file ../datasets/origin/wikicatsum/film/validation.csv \
#	--test_file ../datasets/origin/wikicatsum/film/test.csv \
#	--output_dir ../datasets/ext_oracle/wikicatsum/film_own \
#	--sent_num 30 \
#	--max_sent_len 1024 \
##
##python ./data_preprocess/extractive_oracle_by_generation.py \
#	--dataset scientific_papers \
#	--train_file ../datasets/origin/pubmed/train.csv \
#	--validation_file ../datasets/origin/pubmed/validation.csv \
#	--test_file ../datasets/origin/pubmed/test.csv \
#	--output_dir ../datasets/ext_oracle/pubmed_own \
#	--sent_num 30 \
#	--max_sent_len 1024 \
#	#--add_generated_results False \
#	#--generated_train_result_file ../results/multi_news_own/mle_pretrain/abstractor/bart-base_run1/checkpoint-22000/for_retrieve/train/test_generations.txt \
#	#--generated_validation_result_file ../results/multi_news_own/mle_pretrain/abstractor/bart-base_run1/checkpoint-22000/for_retrieve/val/test_generations.txt \
#	#--generated_test_result_file ../results/multi_news_own/mle_pretrain/abstractor/bart-base_run1/checkpoint-22000/for_retrieve/test/test_generations.txt \
#	#--generated_summ_prefix gen \
#
#python ./data_preprocess/extractive_oracle_by_generation.py \
#	--dataset multi_news \
#	--train_file ../datasets/origin/multi_news/train.csv \
#	--validation_file ../datasets/origin/multi_news/validation.csv \
#	--test_file ../datasets/origin/multi_news/test.csv \
#	--output_dir ../datasets/ext_oracle/with_doc_split_sep/multi_news_1104_f_own \
#	--sent_num 20 \
#	--max_sent_len 1024 \
#	#--add_generated_results False \
#	#--generated_train_result_file ../results/multi_news_own/mle_pretrain/abstractor/bart-base_run1/checkpoint-22000/for_retrieve/train/test_generations.txt \
#	#--generated_validation_result_file ../results/multi_news_own/mle_pretrain/abstractor/bart-base_run1/checkpoint-22000/for_retrieve/val/test_generations.txt \
#	#--generated_test_result_file ../results/multi_news_own/mle_pretrain/abstractor/bart-base_run1/checkpoint-22000/for_retrieve/test/test_generations.txt \
#	#--generated_summ_prefix gen \
#
