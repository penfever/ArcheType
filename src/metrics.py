import json
from statistics import mean

ENDINGS = ["ANSWER:", "CATEGORY:"]

def results_checker_doduo(file_name):
    with open(file_name, "r") as f:
      d = json.load(f)
    correct = 0
    n = len(d)
    per_class_results = dict()
    for k, v in d.items():
        response_set = set(v["response"])
        for r in response_set:
            per_class_results.setdefault(r, {"TP": 0, "FP": 0, "FN": 0, "Total": 0})
        per_class_results.setdefault(v["ground_truth"], {"TP": 0, "FP": 0, "FN": 0, "Total": 0})
        if v['correct'] == True:
            correct += 1
            per_class_results[v["ground_truth"]]["TP"] += 1
        else:
            per_class_results[v["ground_truth"]]["FN"] += 1
            for r in response_set:
                per_class_results[r]["FP"] += 1
        per_class_results[v["ground_truth"]]["Total"] += 1

    for k, v in per_class_results.items():
        v['F1'] = (2 * v["TP"]) / max(1, (2 * v["TP"] + v["FP"] + v["FN"]))

    weighted_f1 = sum([v["F1"] * v["Total"] for k, v in per_class_results.items()]) / n
    unweighted_f1 = mean([v["F1"] for k, v in per_class_results.items()])

    print(f"Total entries: {n} \n Accuracy: {round(correct/n, 4)} \n Weighted F1: {round(weighted_f1, 4)} \n Unweighted F1: {round(unweighted_f1, 4)}")

def results_checker(file_name, skip_duplicates = True):
    with open(file_name, "r") as f:
      d = json.load(f)
    if skip_duplicates:
      d = {k : v for k, v in d.items() if "CATEGORY: *" not in str(k)}

    correct = 0
    good_remap = 0
    total_remap = 0
    truncated = 0
    n = len(d)
    per_class_results = dict()

    for k, v in d.items():
        truncated_flag = True
        for ending in ENDINGS:
            if ending in str(k):
                truncated_flag = False
        if truncated_flag:
            truncated += 1
        per_class_results.setdefault(v["ground_truth"], {"TP": 0, "FP": 0, "FN": 0, "Total": 0})
        per_class_results.setdefault(v["response"], {"TP": 0, "FP": 0, "FN": 0, "Total": 0})
        if v["original_model_answer"] != v["response"]:
            total_remap += 1
            if v['correct'] == True:
                good_remap += 1
        if v['correct'] == True:
            correct += 1
            per_class_results[v["ground_truth"]]["TP"] += 1
        else:
            per_class_results[v["ground_truth"]]["FN"] += 1
            per_class_results[v["response"]]["FP"] += 1
        per_class_results[v["ground_truth"]]["Total"] += 1

    for k, v in per_class_results.items():
        v['F1'] = (2 * v["TP"]) / max(1, (2 * v["TP"] + v["FP"] + v["FN"]))

    weighted_f1 = sum([v["F1"] * v["Total"] for k, v in per_class_results.items()]) / n
    unweighted_f1 = mean([v["F1"] for k, v in per_class_results.items()])

    print(f"Total entries: {n} \n Accuracy: {round(correct/n, 4)} \n Weighted F1: {round(weighted_f1, 4)} \n Unweighted F1: {round(unweighted_f1, 4)} \n Correct Remap: {good_remap} \n Total Remap: {total_remap} \n Truncated: {truncated}")

def missing_entries(f1, f2):
    with open(f1, "r") as file1:
      d1 = json.load(file1)    
    with open(f2, "r") as file2:
      d2 = json.load(file2)
    paths1 = set([v["file+idx"] for _, v in d1.items()])
    paths2 = set([v["file+idx"] for _, v in d2.items()])
    return paths1 - paths2