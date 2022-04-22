import pandas as pd
import argparse
import os
import pdb
import ast
from tqdm import tqdm
#from data_preprocess.merge_generation_results import article_column_mapping

article_column_mapping = {
    "wikicatsum/animal_own": "paragraphs",
    "wikicatsum/company_own": "paragraphs",
    "wikicatsum/film_own": "paragraphs",
    "arxiv_own": "article",
}
summary_column_mapping = {
    "wikicatsum/animal_own": "summary",
    "wikicatsum/company_own": "summary",
    "wikicatsum/film_own": "summary",
    "arxiv_own": "abstract",
}

def remove_empty(df, article_column, summary_column):
    articles = df[article_column]
    cnt = 0
    for i, article in tqdm(enumerate(articles)):
        art_list = ast.literal_eval(article)

        if len(art_list)==0:
            cnt += 1
            df[article_column].iloc[i] = "['Empty input']"
            df["{}_rouge1_f".format(article_column)].iloc[i] = '[0.0]'
            df["{}_rouge2_f".format(article_column)].iloc[i] = '[0.0]'
            df["{}_num_sent".format(article_column)].iloc[i] = 1
            df["{}_num_ext_idx".format(article_column)].iloc[i] = 1

            df["{}_ext".format(summary_column)].iloc[i] = "['Empty input']"
            df["{}_ext_idx".format(summary_column)].iloc[i] = '0'
            df["{}_ext_rouge1_r".format(summary_column)].iloc[i] = '0'
            df["{}_ext_rouge2_r".format(summary_column)].iloc[i] = '0'

        if not df.iloc[i]["summary_gen"]:
            pdb.set_trace()



    print(cnt)
    return df

def main(args):

    df_train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    df_val = pd.read_csv(os.path.join(args.data_dir, "validation.csv"))
    df_test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

    article_column = article_column_mapping[args.dataset_name]
    summary_column = summary_column_mapping[args.dataset_name]
    
    df_val = remove_empty(df_val, article_column, summary_column)
    df_train = remove_empty(df_train, article_column, summary_column)
    df_test = remove_empty(df_test, article_column, summary_column)

    os.makedirs(args.output_dir, exist_ok=True)

    df_val.to_csv(args.output_dir+"/validation.csv", index=False)
    df_train.to_csv(args.output_dir+"/train.csv", index=False)
    df_test.to_csv(args.output_dir+"/test.csv", index=False)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Split the dataset for better parallel processing."
    )
    parser.add_argument("--dataset_name", type=str, required=True, help="name of dataset to check data number")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    main(args) 

