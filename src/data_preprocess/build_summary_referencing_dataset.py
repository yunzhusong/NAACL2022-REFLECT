""" Combining the generated summary to dataset """

import os
import argparse
import pandas as pd


def main(args):


    if args.gen_dir is None and args.gen_train_file and args.gen_validation_file and args.test_file is None:
        ValueError("Please specify args.gen_dir or args.gen_*_file")

    os.makedirs(args.merged_data_dir, exist_ok=True)

    if args.gen_dir is not None:
        # The {split} file should be placed as {args.gen_dir}/{split}_inference/test_generations.csv, where {split} could be train/validation/test
        splits = ["train", "validation", "test"]

        for split in splits:
            dataset_df = pd.read_csv(os.path.join(args.data_dir, split+'.csv'))
            gen_summary_df = pd.read_csv(os.path.join(args.gen_dir, split+"_inference", "test_generations.csv"))
            dataset_df["summary_gen"] = gen_summary_df["summary_gen"]
            dataset_df.to_csv(os.path.join(args.merged_data_dir, split+".csv"), index=False)
            print(f"Combining {args.gen_dir}/{split}_inference and saving to {args.merged_data_dir}/{split}.csv")

    else:
        if args.gen_train_file is not None:
            dataset_df = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
            gen_summary_df = pd.read_csv(args.gen_train_file)
            dataset_df["summary_gen"] = gen_summary_df["summary_gen"]
            dataset_df.to_csv(os.path.join(args.merged_data_dir, "train.csv"), index=False)
            print(f"Combining {args.gen_train_file} and saving to {args.merged_data_dir}")
            
        if args.gen_validation_file is not None:
            dataset_df = pd.read_csv(os.path.join(args.data_dir, 'validation.csv'))
            gen_summary_df = pd.read_csv(args.gen_validation_file)
            dataset_df["summary_gen"] = gen_summary_df["summary_gen"]
            dataset_df.to_csv(os.path.join(args.merged_data_dir, "validation.csv"), index=False)
            print(f"Combining {args.gen_validation_file} and saving to {args.merged_data_dir}")
            
        if args.gen_test_file is not None:
            dataset_df = pd.read_csv(os.path.join(args.data_dir, 'test.csv'))
            gen_summary_df = pd.read_csv(args.gen_test_file)
            dataset_df["summary_gen"] = gen_summary_df["summary_gen"]
            dataset_df.to_csv(os.path.join(args.merged_data_dir, "test.csv"), index=False)
            print(f"Combining {args.gen_test_file} and saving to {args.merged_data_dir}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Split the dataset for better parallel processing."
    )
    parser.add_argument("--merged_data_dir", type=str, required=True, help="Directory path for dataset after combining")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory path for dataset before combining")
    parser.add_argument("--col_name_of_summary_reference", type=str, default="summary_gen", help="Column name of the summary reference")

    # Option 1. Provide the reference information seperating for train/validation/test file
    parser.add_argument("--gen_train_file", type=str, default=None, help="A csv file path with header containing generated summary of train dataset.")
    parser.add_argument("--gen_validation_file", type=str, default=None, help="A csv file path with header containing generated summary of train dataset.")
    parser.add_argument("--gen_test_file", type=str, default=None, help="A csv file path with header containing generated summary of train dataset.")

    # Option 2. Follow the structure to provide summary reference, remember to specify column name
    # |_ $gen_dir\
    #   |_ test_inference\
    #       |_test_generations.csv -> a csv file with header containing summary reference for each data
    #   |_ validation_inference\
    #       |_validation_generations.csv -> a csv file with header containing summary reference for each data
    #   |_ train_inference\
    #       |_train_generations.csv -> a csv file with header containing summary reference for each data
    
    parser.add_argument("--gen_dir", type=str, default=None, help="Directory path where we put the generetaed summary.")

    args = parser.parse_args()
    main(args)
