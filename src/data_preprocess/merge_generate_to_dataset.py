import os
import pandas as pd
import pdb

gen_summary_path = "/home/yunzhu032/Workspace/MDS/results/xscience2_own/mle_pretrain/bart-large/singlestage/checkpoint-1000"
data_path = "/home/yunzhu032/Workspace/MDS/datasets/ext_oracle/with_doc_split_sep/xscience2_own"
new_data_path = "/home/yunzhu032/Workspace/MDS/datasets/ext_oracle/with_doc_split_sep/xscience2_bl_own"

os.makedirs(new_data_path, exist_ok=True)

files = ["train", "validation", "test"]


for file_ in files:
    dataset_df = pd.read_csv(os.path.join(data_path, file_+'.csv'))
    gen_summary_df = pd.read_csv(os.path.join(gen_summary_path, file_+"_inference", "test_generations.csv"))

    dataset_df["summary_gen"] = gen_summary_df["summary_gen"]
    print(dataset_df)
    dataset_df.to_csv(os.path.join(new_data_path, file_+".csv"), index=False)




