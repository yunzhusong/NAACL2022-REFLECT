import os
import re
import math
import argparse
import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm

tqdm.pandas()
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)
import pdb

summarization_name_mapping = {
    "xsum": ("document", "summary"),
    "cnn_dailymail": ("article", "highlights"),
    "newsroom": ("text", "summary"),  # Manual 
    "multi_news": ("document", "summary"),
    "gigaword": ("document", "summary"),
    "wikihow": ("text", "headline"),  # Manual
    "reddit": ("content", "summary"),
    "reddit_tifu": ("documents", "tldr"),
    "big_patent": ("description", "abstract"),
    "scientific_papers": ("article", "abstract"),
    "aeslc": ("email_body", "subject_line"),
    "billsum": ("text", "summary"),
}

##### From PreSumm (EMNLP 2019) ####
# NOTE: this is way more faster than huggingface metrics

def _get_ngrams(n, text):
    """Calcualtes n-grams.
    Args:
      n: which n-grams to calculate
      text: An array of tokens
    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    words = sum(sentences, [])
    return _get_ngrams(n, words)


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


def avg_rouge(file_path,
              art_column,
              summ_column
             ):
    print("Calulating rouge score between {} and {} in {}".format(art_column, summ_column, file_path))
    metric = load_metric("rouge")
    df = pd.read_csv(file_path)

    articles = df[art_column] 
    summaries = df[summ_column]

    result = metric.compute(predictions=summaries,
                            references=articles,
                            use_stemmer=True)
    print(result)
    #result = cal_rouge(summaries, articles)
    return result


def merge_and_split(docs_sent_list, sents):

    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    #abstract = sum(abstract_sent_list, [])
    #abstract = _rouge_clean(' '.join(abstract)).split()

    docs_sents = []
    docs_1grams = []
    docs_2grams = []
    # num of sents for each document
    num_doc_sents = []
    # total num of sents
    num_sents = sum([len(doc_sents) for doc_sents in docs_sent_list])
    # num of documents
    num_docs = len(docs_sent_list)

    for doc in docs_sent_list:
        doc_sents = [_rouge_clean(' '.join(s)).split() for s in doc]
        doc_1grams = [_get_word_ngrams(1, [sent]) for sent in doc_sents]
        doc_2grams = [_get_word_ngrams(2, [sent]) for sent in doc_sents]
 
        docs_sents.append(doc_sents)
        docs_1grams.append(doc_1grams)
        docs_2grams.append(doc_2grams)
        num_doc_sents.append(len(doc_sents))

    score_matrix = np.zeros((num_sents, num_sents))
    for i in range(num_docs):
        for j in range(num_docs):
            if j<=i:
                continue
            
            for k in range(len(docs_sents[i])):
                for l in range(len(docs_sents[j])):
                    
                    x = sum(num_doc_sents[:i]) + k
                    y = sum(num_doc_sents[:j]) + l

                    rouge_1_xy = cal_rouge(docs_1grams[i][k], docs_1grams[j][l])['r']
                    rouge_2_xy = cal_rouge(docs_2grams[i][k], docs_2grams[j][l])['r']
                    rouge_1_yx = cal_rouge(docs_1grams[j][l], docs_1grams[i][k])['r']
                    rouge_2_yx = cal_rouge(docs_2grams[j][l], docs_2grams[i][k])['r']

                    rouge_xy = rouge_1_xy + rouge_2_xy
                    rouge_yx = rouge_1_yx + rouge_2_yx

                    if rouge_xy>rouge_yx:
                        score_matrix[y,x] = rouge_xy
                    else:
                        score_matrix[x,y] = rouge_yx


    max_index = np.argmax(score_matrix, axis=1)
    max_matrix = np.zeros(score_matrix.shape)
    max_matrix[np.arange(num_sents), max_index] = 1

    threshold = score_matrix>0.5

    matrix = threshold * max_matrix
    unwanted = matrix.sum(axis=1)

    sents_tokens = [s for doc in docs_sent_list for s in doc] 
    merged_sents = []
    merged_sents_tokens = []
    for i in range(num_sents):
        if unwanted[i]!=1:
            merged_sents.append(sents[i])
            merged_sents_tokens.append(sents_tokens[i])


    cnt = 0
    merged_docs_sents = []
    for num_sents in num_doc_sents:
        merged_doc_sents = []
        for _ in range(num_sents):
            if unwanted[cnt]!=1:
                merged_doc_sents.append(sents[cnt])
            cnt += 1
        merged_docs_sents.append(merged_doc_sents)

        


    return merged_sents_tokens, merged_sents, merged_docs_sents
    
def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):

    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


##############################


def extractive_oracle(args,
                      file_path,
                      art_column,
                      summ_column,
                      max_samples=None,
                      sent_num=None,
                      max_sent_len=None,
                      for_fusion=None):
    file_dir, file_name = os.path.split(file_path)
    assert args.output_dir != file_dir

    # Read CSV
    df = pd.read_csv(file_path)
    if max_samples:
        df = df.iloc[:max_samples]

    # Create extractive oracle and add back to dataset
    print(f"Creating extractive oracle according to ROUGE for {file_path}...")

    def _extractive_oracle(data):
        if isinstance(data[art_column], str):  # NOTE: emtpy string with be NaN

            # NOTE: handle none input
            if not isinstance(data[art_column], str):
                data[art_column] = " "
            if not isinstance(data[summ_column], str):
                data[summ_column] = " "

            # NOTE: handle special tokens in multi-documents
            if args.dataset == 'multi_news':

                arts, art_sents = [], []
                art_docs_tokens = []
                articles = data[art_column].split("|||||")[:-1]
                for art in articles:
                    if art == "":
                        continue
                    sents = nltk.sent_tokenize(art)
                    art_sents += sents
                    art_docs_tokens.append([nltk.word_tokenize(sent) for sent in sents])
                    # Add <cls> before each sent
                    arts.append('<s>'+'<s>'.join(sents))


                if args.merge_and_split:
                    data["num_sents_before"] = len(art_sents)
                    art_sents_tokens, art_sents, art_docs_sents = merge_and_split(art_docs_tokens, art_sents)
                    arts = ['<s>'+'<s>'.join(doc_sents) for doc_sents in art_docs_sents]
                    data[art_column] = '</s>'.join(arts)[3:]
                else:
                    # Add </s> for split documents
                    data[art_column] = '</s>'.join(arts)[3:]
                    art_sents_tokens = [nltk.word_tokenize(sent) for sent in art_sents]
                    art_sents_tokens = [s[:max_sent_len] for s in art_sents_tokens]
            else:
                art_sents = nltk.sent_tokenize(data[art_column])
                art_sents_tokens = [nltk.word_tokenize(sent) for sent in art_sents]
                art_sents_tokens = [s[:max_sent_len] for s in art_sents_tokens]

            summ_sent = data[summ_column]
            summ_sent_tokens = [nltk.word_tokenize(summ_sent)]

            selected_indices = greedy_selection(art_sents_tokens,
                                                summ_sent_tokens, sent_num)

            if not for_fusion:
                selected_sents = []
                # NOTE: Add sort()
                for idx in np.sort(selected_indices):
                    selected_sents.append(art_sents[idx])

                # Add extractive oracle back to dataset
                data[summ_column + '_ext'] = ' '.join(selected_sents)
                #data[summ_column + '_ext_idx'] = ' '.join([str(idx) for idx in selected_indices])
                data[summ_column + '_ext_idx'] = ' '.join(list(map(str, selected_indices)))
                data[art_column + '_num_sent'] = len(art_sents)

            else:
                selected_sents_1 = []
                selected_sents_2 = []
                binary_flag = True
                for idx in selected_indices:
                    if binary_flag:
                        selected_sents_1.append(art_sents[idx])
                    else:
                        selected_sents_2.append(art_sents[idx])
                    binary_flag = not(binary_flag)

                # Add extractive oracle back to dataset
                data[summ_column + '_ext_1'] = ' '.join(selected_sents_1)
                data[summ_column + '_ext_2'] = ' '.join(selected_sents_2)
        else:
            if not for_fusion:
                # NOTE: use summary sentence if the article is empty
                data[summ_column + '_ext'] = data[summ_column]
                data[summ_column + '_ext_idx'] = '0'
            else:
                data[summ_column + '_ext_1'] = data[summ_column]
                data[summ_column + '_ext_2'] = data[summ_column]

        return data


    #df = df.progress_apply(lambda d: _extractive_oracle(d), axis=1)
    df = df.parallel_apply(lambda d: _extractive_oracle(d), axis=1)

    # Output results as CSV
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, file_name), index=False)


def main(args, art_column, summ_column):

    extractive_oracle(args, args.validation_file, art_column, summ_column,
                      args.max_val_samples, args.sent_num, args.max_sent_len,
                      args.for_fusion)
    extractive_oracle(args, args.train_file, art_column, summ_column,
                      args.max_train_samples, args.sent_num, args.max_sent_len,
                      args.for_fusion)
    extractive_oracle(args, args.test_file, art_column, summ_column,
                      args.max_test_samples, args.sent_num, args.max_sent_len,
                      args.for_fusion)

    
    all_result = {}
    all_result["test"] = avg_rouge(os.path.join(args.output_dir, "test.csv"),
                                   art_column="summary_ext",
                                   summ_column="summary")

    all_result["train"] = avg_rouge(os.path.join(args.output_dir, "train.csv"),
                                   art_column="summary_ext",
                                   summ_column="summary")

    all_result["validation"] = avg_rouge(os.path.join(args.output_dir, "validation.csv"),
                                   art_column="summary_ext",
                                   summ_column="summary")

    with open(os.path.join(args.file_path, "rouge.txt"), "w") as f:
        f.write("The Rouge score between 'summary_ext' and 'summary'")
        f.write(json.dumps(all_result, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract Extractive Oracle according to ROUGE')
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--validation_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_train_samples", type=int, default=1000000)
    parser.add_argument("--max_val_samples", type=int, default=1000000)
    parser.add_argument("--max_test_samples", type=int, default=1000000)
    parser.add_argument("--sent_num", type=int, default=3)
    parser.add_argument("--max_sent_len", type=int, default=128)
    parser.add_argument("--for_fusion", action='store_true')
    parser.add_argument("--merge_and_split", type=bool, default=False)
    args = parser.parse_args()

    art_column, summ_column = summarization_name_mapping.get(args.dataset, None)

    main(args, art_column, summ_column)


'''
# Use huggingface metric, but slower
import datasets
from datasets import load_metric
datasets.logging.set_verbosity_error()
metric = load_metric("rouge")

def _extractive_oracle(data):
    cand_ids = list(range(len(art_sents)))
    cand_scores = []
    selected_ids = []
    selected_sents = []
    max_score = -1
    oracle_sent_num = 3
    for _ in range(oracle_sent_num):
        # Compute score for each article sentence
        for cand_id in cand_ids:
            cat_sent = " ".join(selected_sents + [art_sents[cand_id]])
            results = metric.compute(predictions=[cat_sent],
                                     references=[summ_sent],
                                     rouge_types=["rouge1","rouge2"])

            rouge_1 = results['rouge1'].mid.fmeasure
            rouge_2 = results['rouge2'].mid.fmeasure
            cand_scores.append(rouge_1 + rouge_2)

        # Check if improve Rouge
        cur_max_score = max(cand_scores)
        if cur_max_score > max_score:
            max_score = cur_max_score
        else:
            break

        # Record and pop out the selected sentence
        selected_id = cand_ids[cand_scores.index(cur_max_score)]
        selected_ids.append(selected_id)
        selected_sents.append(art_sents[selected_id])

        cand_ids.pop(cand_scores.index(cur_max_score))

        # Reset candidate scores
        cand_scores = []

    # Add extractive oracle back to dataset
    data[summ_column+'_ext'] = ' '.join(selected_sents)

    return data
'''
