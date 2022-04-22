import os
import pdb
import pandas as pd
from tqdm import tqdm

from datasets import load_metric
import argparse


def split_raw_data(args):

    sc = args.size_chunk
    df_list = []
    #data_files = os.listdir(args.data_dir)
    data_files = ["validation.csv", "train.csv", "test.csv",
                  #"challenge_test_abstractivity_0.csv",
                  #"challenge_test_topic_diversity_0.csv",

                  #"challenge_test_abstractivity_1.csv",
                  #"challenge_test_topic_diversity_1.csv",

                  #"challenge_test_abstractivity_2.csv",
                  #"challenge_test_topic_diversity_2.csv",

                  #"challenge_test_abstractivity_3.csv",
                  #"challenge_test_topic_diversity_3.csv",

                  #"challenge_test_abstractivity_4.csv",
                  #"challenge_test_topic_diversity_4.csv",

                  #"challenge_test_abstractivity_5.csv",
                  #"challenge_test_topic_diversity_5.csv",

                  #"challenge_test_abstractivity_6.csv",
                  #"challenge_test_topic_diversity_6.csv",

                  #"challenge_test_abstractivity_7.csv",
                  #"challenge_test_topic_diversity_7.csv",

                  #"challenge_test_abstractivity_8.csv",
                  #"challenge_test_topic_diversity_8.csv",
                 ]
    os.makedirs(args.output_dir, exist_ok=True)

    print("Reading data ...")
    for data_file in data_files:
        df_list.append(pd.read_csv(os.path.join(args.data_dir, data_file)))

    df = pd.concat(df_list)
    num_chunk = (len(df) // sc) + 1
    for i in tqdm(range(num_chunk), desc="Splitting data"):
        df.iloc[i*sc:(i+1)*sc].to_csv(os.path.join(args.output_dir, '{}.csv'.format(i)), index=False)

    print("Split the data files in {} and store in {}"
          .format(args.data_dir, args.output_dir))
    print("Total size: {}, chunk size: {}, num of chunk: {}"
          .format(len(df), args.size_chunk, num_chunk))


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Split the dataset for better parallel processing."
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--size_chunk", type=int, required=2000)
    args = parser.parse_args()

    split_raw_data(args)
    

