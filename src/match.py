from sentence_transformers import util
import openai
import pandas as pd
import re

try:
  from .const import INTEGER_SET, BOOLEAN_SET
  from .data import get_schema_df, fix_labels
except ImportError:
  from const import INTEGER_SET, BOOLEAN_SET
  from data import get_schema_df, fix_labels

def ans_contains_gt(ans_n, fixed_labels):
    for fixed_label in fixed_labels:
      if fixed_label in ans_n:
        # print(f"Fuzzy label {ans_n} contains gt label {fixed_label}: MATCH \n")
        ans_n = fixed_label
        return ans_n
    return None

def gt_contains_ans(ans_n, fixed_labels):
    if ans_n == "":
        return None
    for fixed_label in fixed_labels:
      if ans_n in fixed_label:
        # print(f"GT label {fixed_label} contains fuzzy label {ans_n}: MATCH \n")
        ans_n = fixed_label
        return ans_n
    return None

def basic_contains(ans_n, fixed_labels, method):
    #TODO: not sure the order should be fixed like this, could be made flexible
    if ans_n in fixed_labels:
        return ans_n
    if "ans_contains_gt" in method:
        res = ans_contains_gt(ans_n, fixed_labels)
        if res:
            return res
    if "gt_contains_ans" in method:
        res = gt_contains_ans(ans_n, fixed_labels)
        if res:
            return res
    return None

def get_base_dtype(context):
    dtype = "integer"
    for item in context:
        if not all(char in INTEGER_SET for char in item):
          return "other"
        try:
            if item.endswith(".0") or item.endswith(",0"):
              item = item[:-2]
              item = str(int(item))
            if item.endswith(".00") or item.endswith(",00"):
              item = item[:-3]
              item = str(int(item))
        except:
            return "float"
        temp_item = re.sub(r"[^a-zA-Z0-9.]", "", item)
        if not temp_item.isdigit():
          dtype = "float"
    return dtype    

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
      return lbl
    #Action refers generally to website actions  
    if any(["https://schema.org/CommentAction" in s, "https://schema.org/ViewAction" in s, "https://schema.org/LikeAction" in s, "https://schema.org/InsertAction" in s]):
      lbl = fix_labels("Action", lsd)
      return lbl
    #Photograph is usually a web url which ends in image file extension, other urls do not end in image file extension
    if any([
      s.startswith("https://"), s.startswith("http://")
    ]):
      if any([
      s.endswith(".jpg"), s.endswith(".png"), s.endswith(".jpeg"), s.endswith(".gif"), s.endswith(".svg"), s.endswith(".bmp")
      ]):
        lbl = fix_labels("Photograph", lsd)
        return lbl
      else:
        lbl = fix_labels("url", lsd)
        return lbl
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
      return lbl
    return False

def apply_basic_rules(context, lbl, lsd):
  if not context:
    return lbl
  if not isinstance(context, list):
    return lbl
  schema_df = get_schema_df()
  schema_ids = schema_df["id"].tolist()
  try:
    for s in context:
      s = str(s)
      if any([s.startswith(s1) for s1 in ["OC_", "SRC", "std:", "mean:", "mode:", "median:", "max:", "min:"]]):
          continue
      special_case = run_special_cases(s, lbl, lsd)
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
        return lbl
    if all(("ATC_" in s) for s in context):
      lbl = "concept broader term"
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
    if lbl and "openopen" in lbl:
      lbl = "openinghours"
    if all(s in BOOLEAN_SET for s in context):
      lbl = "boolean"
    if isinstance(lbl, str):
      lbl = fix_labels(lbl, lsd)
    return lbl
  except Exception as e:
    print(f"Exception {e} in apply_basic_rules with context {context}")
    return lbl