import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

from datasets import load_metric

import pdb

def build_extractor_rl_baseline_from_abs_result(data_file, baseline_file, out_file):
    os.makedirs(out_file, exist_ok=True)

    rougeL = np.load(os.path.join(baseline_file, 'train_generations_rouge.npy'))
    df = pd.read_csv(data_file+'/train.csv')
    df['rougeL'] = rougeL
    df.to_csv(out_file+'/train.csv', index=False)

    df = pd.read_csv(data_file+'/validation.csv')
    df['rougeL'] = np.zeros(len(df))
    df.to_csv(out_file+'/validation.csv', index=False)
    
    df = pd.read_csv(data_file+'/test.csv')
    df['rougeL'] = np.zeros(len(df))
    df.to_csv(out_file+'/test.csv', index=False)
    

def build_extractor_rl_baseline_from_oracle_input(data_file, out_file):
    os.makedirs(out_file, exist_ok=True)

    metric = load_metric("rouge")
    rouge_type = "rougeL"

    # train
    df = pd.read_csv(data_file+'/train.csv')
    summary = df["summary"]
    summary_ext = df["summary_ext"]
    rewards = [r.fmeasure for r in metric.compute(predictions=summary_ext, references=summary,
                                                  use_agregator=False, rouge_types=[rouge_type])[rouge_type]]
    df["summary_ext_rougeL"] = rewards
    df.to_csv(out_file+'/train.csv', index=False)

    # validation
    df = pd.read_csv(data_file+'/validation.csv')
    summary = df["summary"]
    summary_ext = df["summary_ext"]
    rewards = [r.fmeasure for r in metric.compute(predictions=summary_ext, references=summary,
                                                  use_agregator=False, rouge_types=[rouge_type])[rouge_type]]
    df["summary_ext_rougeL"] = rewards
    df.to_csv(out_file+'/validation.csv', index=False)

    # test
    df = pd.read_csv(data_file+'/test.csv')
    summary = df["summary"]
    summary_ext = df["summary_ext"]
    rewards = [r.fmeasure for r in metric.compute(predictions=summary_ext, references=summary,
                                                  use_agregator=False, rouge_types=[rouge_type])[rouge_type]]
    df["summary_ext_rougeL"] = rewards
    df.to_csv(out_file+'/test.csv', index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rouge_file")
    parser.add_argument("--data_file")
    parser.add_argument("--out_file")
    args = parser.parse_args()

    #build_extractor_rl_baseline_from_abs_result(args.data_file, args.rouge_file, args.out_file)
    build_extractor_rl_baseline_from_oracle_input(args.data_file, args.out_file)


