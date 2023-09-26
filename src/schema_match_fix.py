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

def run_special_cases(s, d, lsd):
    #EducationalOccupationalCredential is a list of job requirements
    if any([
      "High School diploma or equivalent" in s,
      "Bachelor's degree or equivalent" in s,
      "Master's degree or equivalent" in s,
      "Doctoral degree or equivalent" in s,
    ]):
      lbl = fix_labels("EducationalOccupationalCredential", lsd)
      d['response'] = lbl
      d['correct'] = (d['response'] == d['ground_truth'])
      return d
    #Photograph is usually a web url which ends in image file extension
    if any([
      s.startswith("https://"), s.startswith("http://")
    ]) and \
    any([
      s.endswith(".jpg"), s.endswith(".png"), s.endswith(".jpeg"), s.endswith(".gif"), s.endswith(".svg"), s.endswith(".bmp")
    ]):
      lbl = fix_labels("Photograph", lsd)
      d['response'] = lbl
      d['correct'] = (d['response'] == d['ground_truth'])
      return d
    #Action refers generally to website actions  
    if any(["https://schema.org/CommentAction" in s, "https://schema.org/ViewAction" in s, "https://schema.org/LikeAction" in s, "https://schema.org/InsertAction" in s]):
      lbl = fix_labels("Action", lsd)
      d['response'] = lbl
      d['correct'] = (d['response'] == d['ground_truth'])
      return d
    #ItemList is actually just recipe steps
    if any([
      "whisk" in s.lower(),
      "preheat oven" in s.lower(),
      "pre-heat oven" in s.lower(),
      "remove from oven" in s.lower(),
      "heat non-stick pan" in s.lower(),
      "serve hot" in s.lower(),
      "Let stand" in s.lower(),
    ]):
      lbl = fix_labels("ItemList", lsd)
      d['response'] = lbl
      d['correct'] = (d['response'] == d['ground_truth'])
      return d
    return False

def run_comprehensive_cases(context, lsd):
  lbl = None
  if all(s.endswith(" g") for s in context):
    lbl = "weight"
  if all(s.endswith(" kg") for s in context):
    lbl = "weight"
  if all(s.endswith(" lb") for s in context):
    lbl = "weight"
  if all(s.endswith(" lbs") for s in context):
    lbl = "weight"
  if all(s.endswith(" pounds") for s in context):
    lbl = "weight"
  if all(s.endswith(" cal") for s in context):
    lbl = "calories"
  if all(s.endswith(" kcal") for s in context):
    lbl = "calories"
  if all(s.endswith(" calories") for s in context):
    lbl = "calories"
  if all("review" in s.lower() for s in context):
    lbl = "review"
  if all("recipe" in s.lower() for s in context):
    lbl = "recipe"
  if all(s in BOOLEAN_SET for s in context):
    lbl = "boolean"
  if lbl:
    return fix_labels(lbl, lsd)
  else:
    return None

def schema_match_fix(d, schema_df, lsd):
    context = d["context"]
    schema_ids = schema_df["id"].tolist()
    cc_res = run_comprehensive_cases(context, lsd)
    if cc_res:
        d['response'] = cc_res
        d['correct'] = (d['response'] == d['ground_truth'])
        return d
    for s in context:
        if any([s.startswith(s1) for s1 in ["OC_", "SRC", "std:", "mean:", "mode:", "median:", "max:", "min:"]]):
            continue
        special_case = run_special_cases(s, d, lsd)
        if special_case:
          return special_case
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
    parser.add_argument("--apply_fix", type=bool, help="Apply schema match fix", default=True)
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
    if args.apply_fix:
      for key in tqdm(prompt_dict.keys(), total=len(prompt_dict.keys())):
          prompt_dict[key] = schema_match_fix(prompt_dict[key], schema_df, label_set)
      with open(args.save_path, 'w', encoding='utf-8') as my_f:
          json.dump(prompt_dict, my_f, ensure_ascii=False, indent=4)
    print("Updated scores: \n", results_checker(args.save_path, skip_duplicates = False))