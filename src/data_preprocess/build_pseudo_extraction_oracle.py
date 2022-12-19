import os
import re
import argparse
import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm
from data.build_datasets import summarization_name_mapping

tqdm.pandas()
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)
import pdb

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

def greedy_selection(doc_sent_list, abstract_sent_list, summary_size, rouge_type='f'):

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
    cur_max_rouge_1, cur_max_rouge_2 = 0, 0
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
            rouge_1 = cal_rouge(candidates_1, reference_1grams)[rouge_type]
            rouge_2 = cal_rouge(candidates_2, reference_2grams)[rouge_type]
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_max_rouge_1 = rouge_1
                cur_max_rouge_2 = rouge_2
                cur_id = i
        if (cur_id == -1):
            return sorted(selected), cur_max_rouge_2, cur_max_rouge_2
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected), cur_max_rouge_1, cur_max_rouge_2

def get_rouge_for_each_sentence(doc_sent_list, abstract_sent_list, rouge_type='f'):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    rouge_1_scores = []
    rouge_2_scores = []
    for i in range(len(sents)):
        candidates_1 = [evaluated_1grams[i]]
        candidates_1 = set.union(*map(set, candidates_1))
        candidates_2 = [evaluated_2grams[i]]
        candidates_2 = set.union(*map(set, candidates_2))
        rouge_1 = cal_rouge(candidates_1, reference_1grams)[rouge_type]
        rouge_2 = cal_rouge(candidates_2, reference_2grams)[rouge_type]

        rouge_1_scores.append(round(rouge_1, 4))
        rouge_2_scores.append(round(rouge_2, 4))

    return rouge_1_scores, rouge_2_scores


####################################

def extractive_oracle(args,
                      file_path,
                      art_column,
                      summ_column,
                      df,
                      rouge_type='r',
                      max_samples=None,
                      sent_num=None,
                      max_sent_len=None,
                     ):

    # Create extractive oracle and add back to dataset
    print(f"Creating extractive oracle according to ROUGE for {args.dataset}...")

    if df is None:
        df = pd.read_csv(file_path)
        file_dir, file_name = os.path.split(file_path)
        assert args.output_dir != file_dir

    if max_samples:
        df = df.iloc[:max_samples]

    proper_end_characters = ['.', '?', '!', '...']

    def _extractive_oracle(data):

        if not isinstance(data[art_column], str):
            print("[Warning] There is an empty summary in dataset, gonna replace it with 'Empty content'")
            data[art_column] = "Empty content"
        if not isinstance(data[summ_column], str):
            print("[Warning] There is an empty document in dataset, gonna replace it with 'Empty content'")
            data[summ_column] = "Empty content"

        # NOTE: handle special tokens in multi-documents
        if args.dataset == 'multi_news':

            arts, art_sents = [], []
            #articles = data[art_column].split("|||||")[:-1] # NOTE: multi-news is updated
            articles = data[art_column].split("|||||")

            for art in articles:
                art = art.strip()
                if art == "":
                    continue
                if art[-1] not in proper_end_characters:
                    art = art + '.'

                sents = nltk.sent_tokenize(art)
                art_sents += sents
                arts.append(sents)

            data["document"] = arts

        elif args.dataset == "xscience":
            arts, art_sents = [], []
            articles = [data[art_column]]
            ref_abstracts = data["ref_abstract"][art_column]
            ref_cites = data["ref_abstract"]["cite_N"]
            ref_abstracts_with_cites = []
            for cite, abst in zip(ref_cites, ref_abstracts):
                ref_abstracts_with_cites.append("{} {}".format(cite, abst))
            articles += ref_abstracts_with_cites
            for art in articles:
                art = art.strip()
                if art == "":
                    continue
                if art[-1] not in proper_end_characters:
                    art = art + '.'

                sents = nltk.sent_tokenize(art)
                art_sents += sents
                arts.append(sents)

            data["document"] = arts

        else:
            art_sents = nltk.sent_tokenize(data[art_column])

        # NOTE: Extract for summary

        summ_sent = data[summ_column]

        art_sents_tokens = [nltk.word_tokenize(sent) for sent in art_sents]
        art_sents_tokens = [s[:max_sent_len] for s in art_sents_tokens]
        summ_sent_tokens = [nltk.word_tokenize(summ_sent)]
        max_sent_num = max(sent_num, int(len(art_sents_tokens)*0.3)) # 20, 0.3 for multinews
        selected_indices, selected_rouge_1, selected_rouge_2 = greedy_selection(
            art_sents_tokens,
            summ_sent_tokens,
            max_sent_num,
            rouge_type=rouge_type,
        )

        rouge_1_scores, rouge_2_scores = get_rouge_for_each_sentence(
            art_sents_tokens,
            summ_sent_tokens,
            rouge_type='r',
        )
        selected_sents = []
        for idx in np.sort(selected_indices):
            selected_sents.append(art_sents[idx])

        #summ_column = "summary"
        #art_column = "document"
        data['summary_ext'] = selected_sents # NOTE
        data['summary_ext_idx'] = ' '.join(list(map(str, selected_indices)))
        data['summary_ext_rouge1_r'] = selected_rouge_1
        data['summary_ext_rouge2_r'] = selected_rouge_2

        data['document_num_sent'] = len(art_sents)
        data['document_num_ext_idx'] = len(selected_indices)
        data['document_rouge1_f'] = rouge_1_scores
        data['document_rouge2_f'] = rouge_2_scores
        #data["document"] = art_sents

        """
        #data[summ_column + '_ext'] = ' '.join(selected_sents) # NOTE
        data[summ_column + '_ext'] = selected_sents # NOTE
        data[summ_column + '_ext_idx'] = ' '.join(list(map(str, selected_indices)))
        data[summ_column + '_ext_rouge1_r'] = selected_rouge_1
        data[summ_column + '_ext_rouge2_r'] = selected_rouge_2

        data[art_column + '_num_sent'] = len(art_sents)
        data[art_column + '_num_ext_idx'] = len(selected_indices)
        data[art_column + '_rouge1_f'] = rouge_1_scores
        data[art_column + '_rouge2_f'] = rouge_2_scores
        data[art_column] = art_sents
        """
        return data

    #df = df.progress_apply(lambda d: _extractive_oracle(d, art_summ, sent_summ, ), axis=1)
    df = df.progress_apply(lambda d: _extractive_oracle(d), axis=1)
    #df = df.parallel_apply(lambda d: _extractive_oracle(d), axis=1)

    # Output results as CSV
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, file_name), index=False)


def combine_generated_result_to_df(args,
                                   file_path,
                                   generated_result_file,
                                   generated_summ_column
                                  ):

    with open(generated_result_file) as f:
        generated_summs = f.read().split('\n')

    df = pd.read_csv(file_path)
    df[generated_summ_column] = generated_summs
    return df


def main(args, art_column, summ_column):

    val_df, train_df, test_df = None, None, None
    if args.dataset=="xscience":
        datasets = load_dataset("multi_x_science_sum")
        val_df = pd.DataFrame(datasets["validation"])
        test_df = pd.DataFrame(datasets["test"])
        train_df = pd.DataFrame(datasets["train"])


    if args.validation_file is not None:
        extractive_oracle(args, args.validation_file, art_column, summ_column, val_df,
                          'r', args.max_val_samples, args.sent_num, args.max_sent_len,
                         )
    if args.train_file is not None:
        extractive_oracle(args, args.train_file, art_column, summ_column, train_df,
                         'r', args.max_train_samples, args.sent_num, args.max_sent_len,
                         )
    if args.test_file is not None:
        extractive_oracle(args, args.test_file, art_column, summ_column, test_df,
                          'r', args.max_test_samples, args.sent_num, args.max_sent_len,
                         )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract Extractive Oracle according to ROUGE')
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--sent_num", type=int, default=30)
    parser.add_argument("--max_sent_len", type=int, default=1024)
    args = parser.parse_args()

    art_column, summ_column = summarization_name_mapping.get(args.dataset, None)

    main(args, art_column, summ_column)

