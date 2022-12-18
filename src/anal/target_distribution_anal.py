import os
import re
import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm

tqdm.pandas()
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)
import pdb

from .data_preprocess.extraction import cal_rouge
from .data.build_datasets import summarization_name_mapping 


def examine_target_distribution(file_path,
                                art_column,
                                summ_column,
                                max_samples
                               ):
    file_dir, file_name = os.path.split()




def main(args, art_column, summ_column):

    examine_target_distribution(args.train_file, art_column, summ_column, max_samples=args.max_samples):
    examine_target_distribution(args.validation_file, art_column, summ_column, max_samples=args.max_samples):
    examine_target_distribution(args.test_file, art_column, summ_column, max_samples=args.max_samples):


if __name__='__main__':
    parser.argparse.ArgumentParser(
        description='Examine the target distribution'
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--validation_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)

    parser.add_argument("--max_samples", type=int, default=10000)
    args = parser.parse_args()


    art_column, summ_column = summarization_name_mapping.get(args.dataset, None)

    main(args, art_column, summ_column)
