
# multi_news
python ./data_download/output_dataset.py\
  --output_dir ../datasets/origin/multi_news\
  --dataset_name multi_news\

# xscience
python ./data_download/output_dataset.py\
  --output_dir ../datasets/origin/xscience\
  --dataset_name multi_x_science_sum\

# wiki_cat_sum_animal
python ./data_download/output_dataset.py\
  --output_dir ../datasets/origin/wikicatsum/animal\
  --dataset_name GEM/wiki_cat_sum\
  --dataset_config animal\
  
# wiki_cat_sum_company
python ./data_download/output_dataset.py\
  --output_dir ../datasets/origin/wikicatsum/company\
  --dataset_name GEM/wiki_cat_sum\
  --dataset_config company\
  
# wiki_cat_sum_film
python ./data_download/output_dataset.py\
  --output_dir ../datasets/origin/wikicatsum/film\
  --dataset_name GEM/wiki_cat_sum\
  --dataset_config film\
