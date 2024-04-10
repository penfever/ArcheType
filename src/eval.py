import pandas as pd
import argparse
import json
import os
from tqdm.auto import tqdm

from src.const import INTEGER_SET, BOOLEAN_SET
from src.data import get_schema_df, fix_labels, get_lsd
from src.metrics import results_checker, results_checker_doduo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Takes input parameters for the 'eval' function.")
    parser.add_argument("--input_path", type=str, help="Input path", required=True)
    parser.add_argument("--label_set", type=str, help="Name of label set (SOTAB-91, SOTAB-55, SOTAB-27, D4-ZS, D4-DoDuo, custom)", required=True)
    parser.add_argument("--naive_score", action="store_true", help="Use naive scoring (i.e. original model answer)", default=False)
    parser.add_argument("--confusion_matrix", action="store_true", help="Show confusion matrix", default=False)
    parser.add_argument("--ignore_classes", type=str, help="Classes to ignore, comma sep.", default="")
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
    schema_df = get_schema_df()
    if args.ignore_classes:
      set_of_classes = set()
      args.ignore_classes = args.ignore_classes.split(", ")
      new_prompt_dict = {}
      for k, v in prompt_dict.items():
        gt_class = v["original_label"]
        if gt_class not in args.ignore_classes:
          new_prompt_dict[k] = v
          set_of_classes.add(v["ground_truth"])
      temp_path = "temp.json"
      with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(new_prompt_dict, f, indent=2)
      args.input_path = temp_path
      print("Updated problem size: {} classes".format(len(set_of_classes)))
    print("Updated scores: \n", results_checker(args.input_path, skip_duplicates = False, naive_score = args.naive_score, confusion_matrix = args.confusion_matrix))