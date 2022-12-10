export CUDA_VISIBLE_DEVICES=0

# multi_news
python ./data_preprocess/build_pseudo_extraction_oracle.py \
	--dataset multi_news \
	--test_file ../datasets/origin/multi_news/test.csv \
	--train_file ../datasets/origin/multi_news/train.csv \
	--validation_file ../datasets/origin/multi_news/validation.csv \
	--output_dir ../datasets/ext_oracle/multi_news \
	--sent_num 20 \

# wiki_cat_sum_animal
#python ./data_preprocess/build_pseudo_extraction_oracle.py \
#	--dataset wiki_cat_sum \
#	--train_file ../datasets/origin/wikicatsum/animal/train.csv \
#	--validation_file ../datasets/origin/wikicatsum/animal/validation.csv \
#	--test_file ../datasets/origin/wikicatsum/animal/test.csv \
#	--output_dir ../datasets/ext_oracle/wikicatsum/animal \
#	--sent_num 30 \
#
# wiki_cat_sum_company
#python ./data_preprocess/build_pseudo_extraction_oracle.py \
#	--dataset wiki_cat_sum \
#	--train_file ../datasets/origin/wikicatsum/company/train.pkl \
#	--validation_file ../datasets/origin/wikicatsum/company/validation.pkl \
#	--test_file ../datasets/origin/wikicatsum/company/test.pkl \
#	--output_dir ../datasets/ext_oracle/wikicatsum/company \
#	--sent_num 30 \

# wiki_cat_sum_film
#python ./data_preprocess/build_pseudo_extraction_oracle.py \
#	--dataset wiki_cat_sum \
#	--train_file ../datasets/origin/wikicatsum/film/train.csv \
#	--validation_file ../datasets/origin/wikicatsum/film/validation.csv \
#	--test_file ../datasets/origin/wikicatsum/film/test.csv \
#	--output_dir ../datasets/ext_oracle/wikicatsum/film \
#	--sent_num 30 \

# xscience
#python ./data_preprocess/build_pseudo_extraction_oracle.py \
#	--dataset multi_news \
#	--output_dir ../datasets/ext_oracle/xscience \
#	--sent_num 20 \
#
