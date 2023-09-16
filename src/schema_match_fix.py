import pandas as pd
import argparse
import json
import os
from tqdm.auto import tqdm

try:
  from .const import INTEGER_SET, BOOLEAN_SET
  from .data import get_schema_df, fix_labels, get_lsd
  from .metrics import results_checker, results_checker_doduo
except ImportError:
  from const import INTEGER_SET, BOOLEAN_SET
  from data import get_schema_df, fix_labels, get_lsd
  from metrics import results_checker, results_checker_doduo

def check_contains(s1, s2):
    if s1 in s2:
        return True
    if s2 in s1:
        return True
    return False

def schema_match_fix(d, schema_df, lsd):
    context = d["context"]
    schema_ids = schema_df["id"].tolist()
    for s in context:
        in_rel = False
        cont_rel = False
        if s in schema_ids:
          in_rel = True
        elif any([sid in s for sid in schema_ids]):
          cont_rel = True
        if in_rel or cont_rel:
          if in_rel:
            ss = schema_df[schema_df['id'] == s]
          elif cont_rel:
            schema_df['cont'] = schema_df['id'].apply(lambda x: check_contains(x, s))
            ss = schema_df[schema_df['cont'] == True]
          enumtype = str(ss.iloc[0]["enumerationtype"])
          if enumtype != "":
            lbl = enumtype.split("/")[-1]
          else:
            lbl = ss["label"].tolist()[0]
          lbl = fix_labels(lbl, lsd)
          d['response'] = lbl
          d['correct'] = (d['response'] == d['ground_truth'])
    return d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Takes input parameters for the 'schema match fix' function.")
    parser.add_argument("--input_path", type=str, help="Input path", required=True)
    parser.add_argument("--save_path", type=str, help="Save path", required=True)
    parser.add_argument("--label_set", type=str, help="Name of label set (SOTAB-91, SOTAB-55, SOTAB-27, D4-ZS, D4-DoDuo, custom)", required=True)
    args = parser.parse_args()
    if os.path.isfile(args.input_path):
        with open(args.input_path, 'r', encoding='utf-8') as f:
            prompt_dict = json.load(f)
    else:
        raise Exception("Input path does not exist.")
    if args.label_set == "custom":
      label_set = {"name" : "custom", "label_set" : args.custom_labels, "dict_map" : {c : c for c in args.custom_labels}, 'abbrev_map' : {c : c for c in args.custom_labels}}
    else:
      label_set = get_lsd(args.label_set)
    for key in tqdm(prompt_dict.keys(), total=len(prompt_dict.keys())):
        schema_df = get_schema_df()
        prompt_dict[key] = schema_match_fix(prompt_dict[key], schema_df, label_set)
    with open(args.save_path, 'w', encoding='utf-8') as my_f:
        json.dump(prompt_dict, my_f, ensure_ascii=False, indent=4)
    print("Updated scores: \n", results_checker(args.save_path, skip_duplicates = False))