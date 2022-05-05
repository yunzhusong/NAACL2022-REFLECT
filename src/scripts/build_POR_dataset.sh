export CUDA_VISIBLE_DEVICES=0

# Multi_News
python ./data_preprocess/build_pseudo_extraction_oracle.py \
	--dataset multi_news \
	--test_file ../datasets/origin/multi_news/test.csv \
	--train_file ../datasets/origin/multi_news/train.csv \
	--validation_file ../datasets/origin/multi_news/validation.csv \
	--output_dir ../datasets/ext_oracle/multi_news_own \
	--sent_num 20 \

# WikiCatSum
#python ./data_preprocess/build_pseudo_extraction_oracle.py \
#	--dataset wiki_cat_sum \
#	--train_file ../datasets/origin/wikicatsum/animal/train.csv \
#	--validation_file ../datasets/origin/wikicatsum/animal/validation.csv \
#	--test_file ../datasets/origin/wikicatsum/animal/test.csv \
#	--output_dir ../datasets/ext_oracle/wikicatsum/animal_own \
#	--sent_num 30 \
#
#python ./data_preprocess/build_pseudo_extraction_oracle.py \
#	--dataset wiki_cat_sum \
#	--train_file ../datasets/origin/wikicatsum/company/train.pkl \
#	--validation_file ../datasets/origin/wikicatsum/company/validation.pkl \
#	--test_file ../datasets/origin/wikicatsum/company/test.pkl \
#	--output_dir ../datasets/ext_oracle/wikicatsum/company_own \
#	--sent_num 30 \

#python ./data_preprocess/build_pseudo_extraction_oracle.py \
#	--dataset wiki_cat_sum \
#	--train_file ../datasets/origin/wikicatsum/film/train.csv \
#	--validation_file ../datasets/origin/wikicatsum/film/validation.csv \
#	--test_file ../datasets/origin/wikicatsum/film/test.csv \
#	--output_dir ../datasets/ext_oracle/wikicatsum/film_own \
#	--sent_num 30 \

# Xscience
#python ./data_preprocess/build_pseudo_extraction_oracle.py \
#	--dataset multi_news \
#	--output_dir ../datasets/ext_oracle/xscience_own \
#	--sent_num 20 \
#
