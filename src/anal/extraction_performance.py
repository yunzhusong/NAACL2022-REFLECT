""" 
Evaluate the extraction performance. 
Unused and to be deprecated.
"""
import os
import argparse
from datasets import load_metric
import pdb


metric = load_metric("rouge")

def _postprocess_text(texts):
    texts = [text.strip() for text in texts]
    texts = ["\n".join(nltk.sent_tokenize(text)) for text in texts]
    return texts

def main():

    data_df = pd.read_csv(args.result_dir)
    preds = data_df[args.summary_ext_pred_column]
    golds = data_df[args.summary_column]
    ext_oracles = data_df[args.summary_ext_column]

    preds = _postprocess_text(preds)
    golds = _postprocess_text(golds)
    pdb.set_trace()

    result = metric.compute(predictions=preds,
                            references=golds,
                            use_stemmer=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--summary_ext_pred_column", type=str)
    parser.add_argument("--summary_column", type=str)
    parser.add_argument("--summary_ext_column", type=str)
    args = parser.parse_args()
    main(args)

