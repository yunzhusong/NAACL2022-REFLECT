import os
import argparse
import json
import nltk
import numpy as np
import pandas as pd
from datasets import load_metric

from random import random
import pdb

def basic_statistic(extraction_dir, gen_column):
    print("Perform basic analysis for ", extraction_dir)
    extraction_file = os.path.join(extraction_dir, "test_extractions.csv")
    df = pd.read_csv(extraction_file)
    
    num_empty = int(np.sum([type(i)==float for i in df[gen_column]]))
    avg_length = round(float(np.mean([len(i.split()) if type(i)!=float else 0 for i in df[gen_column]])),2)

    print("Num Empty: {}/{}".format(num_empty, len(df)))
    print("Avg Length: {:.2f}".format(avg_length))
    
    result_file = os.path.join(extraction_dir, "all_results.json")

    if os.path.isfile(result_file):
        with open(result_file) as f:
            results = json.load(f)
        results["test_ext_num_empty"] = num_empty
        results["test_ext_len"] = avg_length
        
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=4)
            print("Write to ", result_file)
        

        print(results)


def cal_rouge(extraction_dir, gen_column):
    print("Calculating Rouge for ", extraction_dir)
    exp_id = random()
    metric = load_metric("rouge", experiment_id=exp_id)

    def _postprocess_text(texts):
        texts = [text.strip() for text in texts]
        texts = ["\n".join(nltk.sent_tokenize(text)) for text in texts]
        return texts

    extraction_file = os.path.join(extraction_dir, "test_extractions.csv")
    df = pd.read_csv(extraction_file)
    preds = df[gen_column]
    golds = df["summary"]

    preds = _postprocess_text(preds)
    golds = _postprocess_text(golds)

    results_ext = metric.compute(predictions=preds, references=golds, use_stemmer=True)
    results_ext = {"test_ext_"+key: value.mid.fmeasure * 100 for key, value in results_ext.items()}

    result_file = os.path.join(extraction_dir, "all_results.json")
    if os.path.isfile(result_file):
        with open(result_file, "r") as f:
            results = json.load(f)

        results = {**results, **results_ext}
        with open(result_file, "w") as f:
            json.dump(results, f, indent=4)
            print("Write to ", result_file)

        print(results)

    


def main(args):
    if args.do_basic_statistic:
        basic_statistic(args.extraction_dir, args.gen_column)


    if args.do_rouge_calculation:
        cal_rouge(args.extraction_dir, args.gen_column)



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction_dir", type=str)
    parser.add_argument("--do_basic_statistic", action="store_true")
    parser.add_argument("--do_rouge_calculation", action="store_true")
    parser.add_argument("--gen_column", type=str, default="ext_article")
    args = parser.parse_args()
    main(args)


