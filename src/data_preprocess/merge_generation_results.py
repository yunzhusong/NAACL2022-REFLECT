import os
import pdb
import pandas as pd
from tqdm import tqdm
import ast

from datasets import load_metric
import argparse

from data_preprocess.clean_dataset import remove_empty

# NOTE: you need to add a new row for another new dataset

num_dataset_mapping = {
    "wikicatsum/animal_own": {"train": 48234,"validation": 2652,"test": 2757},
    "wikicatsum/company_own": {"train": 54978,"validation": 2981,"test": 2955},
    "wikicatsum/film_own": {"train": 52334,"validation": 2861,"test": 3011},
    "arxiv_own": {"train": 203037,"validation": 6436,"test": 6440}
}
summary_column_mapping = {
    "wikicatsum/animal_own": "summary",
    "wikicatsum/company_own": "summary",
    "wikicatsum/film_own": "summary",
    "arxiv_own": "abstract",
}
article_column_mapping = {
    "wikicatsum/animal_own": "paragraphs",
    "wikicatsum/company_own": "paragraphs",
    "wikicatsum/film_own": "paragraphs",
    "arxiv_own": "article",
}
splited_data_dir_mapping = {
    "wikicatsum/animal_own": "../datasets/animal_own_splited",
    "wikicatsum/company_own": "../datasets/company_own_splited",
    "wikicatsum/film_own": "../datasets/film_own_splited",
    "arxiv_own": "../datasets/arxiv_own_splited",
}
splited_gen_dir_mapping = {
    "wikicatsum/animal_own": "../results_for_rebuttal/wiki_cat_sum_animal_own/finetuned_abs/bart-large-cnn-A/checkpoint-21500",
    "wikicatsum/company_own": "../results_for_rebuttal/wiki_cat_sum_company_own/finetuned_abs/bart-large-A/checkpoint-14250",
    "wikicatsum/film_own": "../results_for_rebuttal/wiki_cat_sum_film_own/finetuned_abs/bart-large-cnn-A/checkpoint-12000",
    "arxiv_own": "../results_for_rebuttal/arxiv_own/finetuned_abs/pegasus-A/",
}
merged_data_dir_mapping = {
    "wikicatsum/animal_own": "../datasets/ext_oracle/wikicatsum/animal_bl_own",
    "wikicatsum/company_own": "../datasets/ext_oracle/wikicatsum/company_bl_own",
    "wikicatsum/film_own": "../datasets/ext_oracle/wikicatsum/film_bl_own",
    "arxiv_own": "../datasets/ext_oracle/arxiv_pag_own_2",
}


# NOTE: Order of spliting: validation.csv/train.csv/test.csv

def merge_generated_result_to_dataset_with_shuffle(args):

    from datasets import load_dataset

    dataset_name = args.dataset_name
    splited_data_dir = splited_data_dir_mapping[dataset_name]
    splited_gen_dir = splited_gen_dir_mapping[dataset_name]
    merged_data_dir = merged_data_dir_mapping[dataset_name]
    summary_column = summary_column_mapping[dataset_name]
    article_column = article_column_mapping[dataset_name]

    os.makedirs(merged_data_dir, exist_ok=True)

    # Read the dataset and shuffle for each chunk
    df_list = []
    file_names = os.listdir(splited_data_dir)
    for i in range(len(file_names)):
        data_files = {"chunk": os.path.join(splited_data_dir, "{}.csv".format(i))}
        #data_files[i] = os.path.join(splited_data_dir, "{}.csv".format(i))
        dataset = load_dataset("csv", data_files=data_files)
        dataset = dataset.shuffle(seed=0)
        df_list.append(pd.DataFrame(dataset["chunk"].to_dict()))
    df_data = pd.concat(df_list)

    # Read the generation results
    df_list = []
    file_names = os.listdir(splited_gen_dir)
    for i in range(len(file_names)):
        data_files = {"chunk": "{}/{}/test_generations.csv".format(splited_gen_dir, i)}
        dataset = load_dataset("csv", data_files=data_files)
        df_list.append(pd.DataFrame(dataset["chunk"].to_dict()))
    df_gen = pd.concat(df_list)

    print("Merge the generated summary to dataset")
    df_data["summary_gen"] = df_gen["summary_gen"]

    num_val = num_dataset_mapping[args.dataset_name]["validation"]
    num_train = num_dataset_mapping[args.dataset_name]["train"]
    num_test = num_dataset_mapping[args.dataset_name]["test"]

    assert(num_train+num_val+num_test == len(df_data))

    df_val = df_data[:num_val]
    df_train = df_data[num_val:num_val+num_train]
    df_test = df_data[num_val+num_train:]

    df_val = remove_empty(df_val, article_column)
    df_train = remove_empty(df_train, article_column)
    df_test = remove_empty(df_test, article_column)

    pdb.set_trace()
    df_val.to_csv(merged_data_dir+"/validation.csv", index=False)
    df_test.to_csv(merged_data_dir+"/test.csv", index=False)
    df_train.to_csv(merged_data_dir+"/train.csv", index=False)

    print("Write to {}".format(merged_data_dir))
    with open(os.path.join(merged_data_dir, "info.txt"), 'a') as f:
        f.write("[Warning] In this dataset, the data order is different from the origin.\n")
        f.write("The summary_gen columns is obtained from {}\n".format(splited_gen_dir))
        f.write("The order is validation/train/test, seed=0 by datasets/load_dataset\n. chunk size=6000 \n")
        f.write("# of train: {} \n".format(len(df_train)))
        f.write("# of valid: {} \n".format(len(df_val)))
        f.write("# of test: {} \n".format(len(df_test)))

        metric = load_metric("rouge")
        res = metric.compute(predictions=df_test["summary_gen"],
                         references=df_test["abstract"])
        results = {}
        for k, v in res.items():
            results[k] = '{:.4f}'.format(res[k].mid.fmeasure)
        f.write(str(results))
    print(results)


def merge_generated_result_to_dataset(args):

    dataset_name = args.dataset_name
    splited_data_dir = splited_data_dir_mapping[dataset_name]
    splited_gen_dir = splited_gen_dir_mapping[dataset_name]
    merged_data_dir = merged_data_dir_mapping[dataset_name]
    summary_column = summary_column_mapping[dataset_name]
    article_column = article_column_mapping[dataset_name]

    print("Merge the generated results from {} and {} to {}".format(splited_gen_dir, splited_data_dir, merged_data_dir))

    df_list_data = []
    df_list_gen = []

    num_chunk = len(os.listdir(splited_data_dir))
    for i in range(num_chunk):
        df_list_data.append(pd.read_csv(splited_data_dir+"/{}.csv".format(i)))
        df_list_gen.append(pd.read_csv(splited_gen_dir+"/{}/test_generations.csv".format(i)))

    df_data = pd.concat(df_list_data)
    df_gen = pd.concat(df_list_gen)

    df_data["summary_gen"] = df_gen["summary_gen"]

    num_val = num_dataset_mapping[args.dataset_name]["validation"]
    num_train = num_dataset_mapping[args.dataset_name]["train"]
    num_test = num_dataset_mapping[args.dataset_name]["test"]

    assert(num_train+num_val+num_test == len(df_data))

    df_val = df_data[:num_val]
    df_train = df_data[num_val:num_val+num_train]
    df_test = df_data[num_val+num_train:]

    df_val = remove_empty(df_val, article_column)
    df_train = remove_empty(df_train, article_column)
    df_test = remove_empty(df_test, article_column)

    os.makedirs(merged_data_dir, exist_ok=True)

    df_val.to_csv(merged_data_dir+"/validation.csv", index=False)
    df_train.to_csv(merged_data_dir+"/train.csv", index=False)
    df_test.to_csv(merged_data_dir+"/test.csv", index=False)

    with open(os.path.join(merged_data_dir, "info.txt"), "w") as f:
        f.write("This dataset is merged from the generated results of \n{}\n and data of \n{}\n\n".format(splited_gen_dir, merged_data_dir))
        f.write("# of train: {} \n".format(len(df_train)))
        f.write("# of valid: {} \n".format(len(df_val)))
        f.write("# of test: {} \n".format(len(df_test)))

        results = {}
        metric = load_metric("rouge")
        res = metric.compute(predictions=df_test["summary_gen"],
                         references=df_test[summary_column])
        for k, v in res.items():
            results[k] = '{:.4f}'.format(res[k].mid.fmeasure)
        f.write(str(results))
    print(results)


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Split the dataset for better parallel processing."
    )
    parser.add_argument("--dataset_name", type=str, required=True, help="name of dataset to check data number")
    args = parser.parse_args()

    merge_generated_result_to_dataset(args)
    

