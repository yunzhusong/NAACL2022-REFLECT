import os
import argparse
import pandas as pd
from datasets import load_metric
import json

import pdb

metric = load_metric("rouge")

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

def index_anal(file_path,
              ext_id_column
              ):
    df = pd.read_csv(file_path)
    index = df[ext_id_column]
    for i in range(len(index)):
        map(int, index[i].split())



def avg_rouge(file_path,
              art_column,
              summ_column
             ):
    print("Calulating rouge score between {} and {} in {}".format(art_column, summ_column, file_path))
    df = pd.read_csv(file_path)

    articles = df[art_column] 
    summaries = df[summ_column]

    result = metric.compute(predictions=summaries,
                            references=articles,
                            use_stemmer=True)
    print(result)
    #result = cal_rouge(summaries, articles)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analze Extracted Oracle according to ROUGE')
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--art_column", type=str, required=True)
    parser.add_argument("--summ_column", type=str, required=True)
    args = parser.parse_args()

    all_result = {}
    all_result["test"] = avg_rouge(os.path.join(args.file_path, "test.csv"),
                                   art_column=args.art_column,
                                   summ_column=args.summ_column)

    #all_result["train"] = avg_rouge(os.path.join(args.file_path, "train.csv"),
    #                               art_column=args.art_column,
    #                               summ_column=args.summ_column)

    #all_result["validation"] = avg_rouge(os.path.join(args.file_path, "validation.csv"),
    #                               art_column=args.art_column,
    #                               summ_column=args.summ_column)


    with open(os.path.join(args.file_path, "rouge.txt"), "w") as f:
        f.write("The Rouge score between 'summary_ext' and 'summary'")
        f.write(json.dumps(all_result, indent=4))


