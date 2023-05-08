from sentence_transformers import util
import openai
import pandas as pd
import re

try:
  from .const import INTEGER_SET, BOOLEAN_SET
except ImportError:
  from const import INTEGER_SET, BOOLEAN_SET

def ans_contains_gt(ans_n, fixed_labels):
    for fixed_label in fixed_labels:
      if fixed_label in ans_n:
        print(f"Fuzzy label {ans_n} contains gt label {fixed_label}: MATCH \n")
        ans_n = fixed_label
        return ans_n
    return None

def gt_contains_ans(ans_n, fixed_labels):
    if ans_n == "":
        return None
    for fixed_label in fixed_labels:
      if ans_n in fixed_label:
        print(f"GT label {fixed_label} contains fuzzy label {ans_n}: MATCH \n")
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

def apply_basic_rules(context, lbl):
  if not context:
    return lbl
  if not isinstance(context, list):
    return lbl
  try:
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
      return lbl
  except Exception as e:
      print(f"Exception {e} in apply_basic_rules with context {context}")
      return lbl