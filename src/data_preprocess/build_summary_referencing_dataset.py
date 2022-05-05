import os
import pandas as pd
import pdb


gen_dir_mapping = {
    "xscience_own": "../outputs/xscience_own/finetuned_abs/bart-largs-A/checkpoint-1000"
}
merged_dir_mapping = {
    "xscience_own": "../datasets/ext_oracle/xscience_bl_own"
}


#gen_summary_path = "/home/yunzhu032/Workspace/MDS/results/xscience2_own/mle_pretrain/bart-large/singlestage/checkpoint-1000"
#data_path = "/home/yunzhu032/Workspace/MDS/datasets/ext_oracle/with_doc_split_sep/xscience2_own"
#new_data_path = "/home/yunzhu032/Workspace/MDS/datasets/ext_oracle/with_doc_split_sep/xscience2_bl_own"

os.makedirs(new_data_path, exist_ok=True)

def main(args):

    if args.gen_dir is not None:
        # The {split} file should be placed as {args.gen_dir}/{split}_inference/test_generations.csv, where {split} could be train/validation/test
        splits = ["train", "validation", "test"]

        for split in splits:
            dataset_df = pd.read_csv(os.path.join(args.data_dir, split+'.csv'))
            gen_summary_df = pd.read_csv(os.path.join(gen_summary_path, split+"_inference", "test_generations.csv"))
            dataset_df["summary_gen"] = gen_summary_df["summary_gen"]
            dataset_df.to_csv(os.path.join(args.merged_data_dir, split+".csv"), index=False)
            print(dataset_df)

    else:
        if args.gen_train_file is not None:
            dataset_df = pd.read_csv(args.data_dir + 'train.csv')
            gen_summary_df = pd.read_csv(args.gen_train_file)
            dataset_df["summary_gen"] = gen_summary_df["summary_gen"]
            dataset_df.to_csv(args.merged_data_dir + "train.csv", index=False)
            print(dataset_df)
            
        if args.gen_validation_file is not None:
            dataset_df = pd.read_csv(args.data_dir + 'validation.csv')
            gen_summary_df = pd.read_csv(args.gen_train_file)
            dataset_df["summary_gen"] = gen_summary_df["summary_gen"]
            dataset_df.to_csv(args.merged_data_dir + "validation.csv", index=False)
            print(dataset_df)
            
        if args.gen_test_file is not None:
            dataset_df = pd.read_csv(args.data_dir + 'test.csv')
            gen_summary_df = pd.read_csv(args.gen_train_file)
            dataset_df["summary_gen"] = gen_summary_df["summary_gen"]
            dataset_df.to_csv(args.merged_data_dir + "test.csv", index=False)
            print(dataset_df)


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Split the dataset for better parallel processing."
    )
    parser.add_argument("--merged_data_dir", type=str, required=True, help="Directory path for merged dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory path for dataset")

    parser.add_argument("--gen_dir", type=str, default=None, help="Directory path where we put the generetaed summary.")
    parser.add_argument("--gen_train_file", type=str, default=None, help="File path of generated summary for train dataset.")
    parser.add_argument("--gen_validation_file", type=str, default=None, help="File path of generated summary for validation dataset.")
    parser.add_argument("--gen_test_file", type=str, default=None, help="File path of generated summary for test dataset.")

    args = parser.parse_args()
    main(args)
