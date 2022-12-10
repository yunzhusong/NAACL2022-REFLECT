import os
from os.path import join as pjoin
import argparse
from datasets import load_dataset
import pandas as pd

import pdb

def output_datasets(args):

    if args.dataset_name is not None:
        # Download and load a dataset from the hub.
        if args.dataset_name == 'cnn_dailymail':
            args.dataset_config_name = '3.0.0'
        elif args.dataset_name == 'reddit_tifu':
            args.dataset_config_name = 'long'
        elif args.dataset_name == 'wikihow':
            args.dataset_config_name = 'all'
        elif args.dataset_name == 'arxiv':
            args.dataset_name = 'scientific_papers'
            args.dataset_config_name = 'arxiv'
        elif args.dataset_name == 'pubmed':
            args.dataset_name = 'scientific_papers'
            args.dataset_config_name = 'pubmed'
        #else:
        #    args.dataset_config_name = None

        if args.dataset_name == 'wikihow':
            datasets = load_dataset(args.dataset_name,
                                    args.dataset_config_name,
                                    data_dir=os.path.abspath("../cache/manual/wikihow"))
        elif args.dataset_name == 'newsroom':
            datasets = load_dataset(args.dataset_name,
                                    args.dataset_config_name,
                                    data_dir=os.path.abspath("../cache/manual/newsroom"))
        else:
            datasets = load_dataset(args.dataset_name,
                                    args.dataset_config_name)
    else:
        # Use your own dataset
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        if args.test_file is not None:
            data_files["test"] = args.test_file
            extension = args.test_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset_name == 'GEM/wiki_cat_sum':

        for key in datasets.keys():
            df = pd.DataFrame.from_dict(datasets[key])

            for col in df.columns:

                if col == 'paragraphs':
                    values = df[col]
                    # wrap the whole paragraph as a single document
                    new_values = [[v] for v in values]
                    #new_values = [' '.join(v) for v in values]
                    df[col] = new_values
    
                if col == 'summary':
                    values = df[col]

                    text_list, topic_list = [], []
                    for i in range(len(values)):
                        text_list.append('\n'.join(values[i]['text']))
                        topic_list.append(values[i]['topic'])
                    df['summary'] = text_list
                    df['topic'] = topic_list

            #df.to_pickle(pjoin(args.output_dir, key+'.pkl'))
            with open(pjoin(args.output_dir, key+'.csv'), 'wb') as f_o:
                df.to_csv(f_o, index=False)
    else:
        for key in datasets.keys():
            #df.to_pickle(pjoin(args.output_dir, key+'.pkl'))
            with open(pjoin(args.output_dir, key+'.csv'), 'wb') as f_o:
                datasets[key].to_csv(f_o, index=False)

    """
    if args.do_predict:
        if "test" not in datasets:
            raise ValueError("--args.do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if args.max_test_samples is not None:
            test_dataset = test_dataset.select(
                range(args.max_test_samples))

        with open(pjoin(args.output_dir,'test.csv'), 'wb') as f_o:
            test_dataset.to_csv(f_o, index=False)
    """

def main(args):
    output_datasets(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Output Dataset into CSV for other usage.')
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--validation_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--do_predict", action='store_true')
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    args = parser.parse_args()

    main(args)
