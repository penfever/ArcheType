import pandas as pd
import json
import argparse

def find_context(s):
    idx = s.find("INPUT:")
    len_c = len("INPUT:")
    s = s[idx + len_c:] + "\n"
    return s

def make_dataset(args):

    with open(args.in_file, 'r') as jf:
        prompt_dict = json.load(jf)

    values = []
    kl = list(prompt_dict.keys())
    for k in kl:
        values.append(prompt_dict[k]['ground_truth'])

    df = pd.DataFrame(list(zip(kl, values)),
                columns =['prompt', 'completion'])

    target_path = args.in_file.replace(".json", "-dataset.json")

    df['instruction'] = "Select the category which best matches the input. \n"
    df['input'] = df['prompt'].apply(lambda x : find_context(x))
    df['output'] = df['completion'] + "\n"
    df = df.drop(columns=['prompt', 'completion'])
    df.to_json(target_path, orient='records', indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Takes input parameters for the 'make dataset' function.")
    parser.add_argument("--in_file", type=str, default="/scratch/bf996/llm_er_std/proj/CTA_CPA_Benchmarks/wotab/train-llama-oc+tn+ss-v1.json", help="The name of the ArcheType output JSON to read data from.")
    args = parser.parse_args()
    make_dataset(args)