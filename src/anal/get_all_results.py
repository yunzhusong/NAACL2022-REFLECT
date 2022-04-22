import os
import re
import pdb
import json
import argparse
import pandas as pd

result_file = "all_results.json"
objs = ['test_rouge1', 'test_rouge2', 'test_rougeL', 'test_rougeLsum',
        'test_ext_rouge1', 'test_ext_rouge2', 'test_ext_rougeL', 'test_ext_rougeLsum',
        'test_ext_f1', 'test_ext_recall', 'test_ext_precision', 'test_ext_gen_len',
        'test_gen_len', 'test_loss', 'test_runtime', 'test_samples',
       ]
#cols = objs + ["exp_name", "ckpt", "exp_path"]
cols = objs + ["ckpt", "exp_path"]
ignore_files = ["merges.txt", "pytorch_model.bin", "config.json",
               "data_args.bin", "model_args.bin", "training_args.bin",
                "vocab.json", "trainer_state.json", "tokenizer_config.json",
                "test_generation.txt", "test_gold.txt",
                "optimizer.pt", "scheduler.pt", "special_tokens_map.json"]

records = {}

def record_obj(args, path_name):


    m = re.search('(?<=checkpoint-)\d+', path_name)
    if m is not None:
        ckpt = m.group(0)
    else:
        ckpt = None

    name = path_name.replace("/checkpoint-{}".format(ckpt), "")
    name = name.replace(args.result_dir, "")

    prefix = '/'.join(name.split('/')[:-5])
    suffix = '/'.join(name.split('/')[-5:])

    output = {
        "exp_path": name,
        #"exp_path": path_name,
        #"exp_name": suffix,
        "ckpt": ckpt
    }

    if os.path.isfile(os.path.join(path_name, result_file)):
        with open(os.path.join(path_name, result_file)) as f:
            results = json.load(f)
            for k, v in results.items():
                if k in objs:
                    output[k] = v

    records[path_name] = output

def search_and_record(args, path_name):
   
    #new_records = []
    if os.path.isdir(path_name):
        dir_names = os.listdir(path_name)
        for dir_name in dir_names:
            #if "checkpoint" in dir_name or "by_bart_large" in dir_name:
            current_path_name = os.path.join(path_name, dir_name)
            if os.path.isdir(current_path_name):
                if result_file in os.listdir(current_path_name):
                    record_obj(args, os.path.join(path_name, dir_name))

                # If checkpoint exists, try to find the test results
                #record_obj(args, os.path.join(path_name, dir_name))
                #records.update(new_records)
                #records[k] = v
                """
                in_dir_names = os.listdir(os.path.join(path_name, dir_name))
                if result_file in in_dir_names:
                    records.append(record_obj(os.path.join(path_name, dir_name)))
                for in_dir_name in in_dir_names:
                    if result_file == in_dir_name:
                        records.append(record_obj(os.path.join(path_name, dir_name)))
                    else:
                        records += search_and_record(os.path.join(path_name, dir_name, in_dir_name))
                """
                #return records

                search_and_record(args, os.path.join(path_name, dir_name))
            #elif dir_name in ignore_files:
            #    continue

            #if os.path.isdir(os.path.join(path_name, dir_name)):
            #    search_and_record(args, os.path.join(path_name, dir_name))
            #else:
            #    # If not a checkpoint folder go inside and search for it.
            #    if os.path.isdir(os.path.join(path_name, dir_name)):
            #        search_and_record(args, os.path.join(path_name, dir_name))
            """
                    in_dir_names = os.listdir(os.path.join(path_name, dir_name))
                    for in_dir_name in in_dir_names:
                        search_and_record(os.path.join(path_name, dir_name, in_dir_name))
                        #records.update(new_records)
            """
def main(args):

    search_and_record(args, args.result_dir)
    df = pd.DataFrame.from_dict(records, orient="index")
    df = df.reset_index()
    df = df[cols]
    print(df)
    
    df.to_csv(args.output_dir)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)


