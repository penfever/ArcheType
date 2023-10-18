import json
from statistics import mean
import numpy as np
import os
import torch
import logging
from datetime import datetime
import time
from pathlib import Path

try:
    from sklearn.metrics import confusion_matrix
    import pandas as pd
    import numpy as np
    import seaborn as sn
    from matplotlib import pylab as plt
except Exception as e:
    logging.warning("Error loading extended metrics libraries: extended metrics will fail")
    logging.warning(e)

ENDINGS = ["ANSWER:", "CATEGORY:"]

def reorder_matrix(mat, showdendrogram = False):
    Y = sch.linkage(mat, method='centroid')
    Z = sch.dendrogram(Y, orientation='left', no_plot= not showdendrogram)

    index = Z['leaves']
    mat = mat[index,:]
    mat = mat[:,index]
    return mat, index
    # Plot distance matrix.

def show_matrix(args, mat, title):
    fig = plt.figure(figsize=(28, 28), dpi=200)
    fig.suptitle(title, ha = "center", fontsize = 20)

    axmatrix = fig.add_axes([0, 0, 0.9,0.9], label='axes1')
    im = axmatrix.matshow(mat, aspect='auto',  origin='lower')
    res = str(datetime.now())[:19]
    res = res.translate({ord(":"): "-", ord(" "):"_"})
    axcolor = fig.add_axes([0.95,0,0.02,0.9])
    plt.colorbar(im, cax=axcolor)
    save_path = os.path.join(conf_path, 'clustered_confusion_matrix_{}.svg'.format(res))
    fig.savefig(save_path, format='svg', dpi=200)

def write_confusion_matrix(log_base_path, file_name, output, labels, classes):
    #confusion matrix
    cf_matrix = confusion_matrix(labels, output)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * 10, index = [i for i in classes],
                        columns = [i for i in classes])
    df_cm = df_cm.div(df_cm.sum(axis=1), axis=0)
    conf_path = os.path.join(log_base_path, f"{file_name}_confusion_matrix")
    if not os.path.exists(conf_path):
        os.mkdir(conf_path)
    res = str(datetime.now())[:19]
    res = res.translate({ord(":"): "-", ord(" "):"_"})
    logging.info('Writing confusion matrix')
    df_cm.to_csv(os.path.join(conf_path, "confusion_matrix_{}.csv".format(res)), index=False)
    per_class_acc = pd.Series(np.diag(df_cm), index=[df_cm.index, df_cm.columns]).round(2)
    per_class_acc = pd.DataFrame(per_class_acc).transpose()
    per_class_acc.columns = [''.join(col[1:]) for idx, col in enumerate(per_class_acc.columns.values)]
    per_class_acc.to_csv(os.path.join(conf_path, "per_class_acc_{}.csv".format(res)), index=False)
    font_size = round(1 * 100//len(classes), 2)
    if font_size < 0.1:
        font_size = 0.1
    sn.set(font_scale=font_size)
    #if len(classes) < 201:
    start = time.time()
    print("Saving confusion matrix: this might take a while...")
    plt.figure(figsize = (168,80), dpi=200)
    sn.heatmap(df_cm, annot=True)
    plt.savefig(os.path.join(conf_path, "confusion_matrix_{}.svg".format(res)), format='svg', dpi=200)        
    plt.close('all')
    print("Saving confusion matrix: done in {} seconds".format(time.time() - start))
    #class-class clustering matrix
    # logging.info('Saving class-class clustering matrix')
    # logit_concat = np.concatenate(output, axis=0)
    # corr_mat_logits = np.corrcoef(logit_concat, rowvar=False)
    # corr_mat_logits[corr_mat_logits < 0] = 0 # not quite necessary, but helps sharpen the blocks
    # try:
    #     corr_mat_logits, indices_logits = reorder_matrix(corr_mat_logits)
    #     show_matrix(args, corr_mat_logits, 'Logit-based Similarity Matrix')
    # except Exception as e:
    #     logging.warning("Clustering matrix did not save")
    #     logging.warning(e)

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

def results_checker(file_name, skip_duplicates = True, naive_score = False, confusion_matrix = False):
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
    if confusion_matrix:
        class_names = set()
        for k, v in d.items():
            class_names.add(v["ground_truth"])
        class_names = sorted(list(class_names))
        logits = np.zeros(n)
        labels = np.zeros(n)
    for idx, (k, v) in enumerate(d.items()):
        if naive_score:
            v['response'] = v['original_model_answer']
            v['correct'] = (v['response'] == v['ground_truth'])
        if confusion_matrix:
            gt_index = class_names.index(v["ground_truth"])
            try:
                pred_index = class_names.index(v["response"])
            except ValueError:
                options = np.arange(0, gt_index) + np.arange(gt_index + 1, len(class_names))
                rand_seed = 0
                pred_index = np.random.choice(options, 1, replace = False)
            logits[idx] = pred_index
            labels[idx] = gt_index
        # truncated_flag = True
        # for ending in ENDINGS:
        #     if ending in str(k):
        #         truncated_flag = False
        # if truncated_flag:
        #     truncated += 1
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

    print(f"Total entries: {n} \n Accuracy: {round(correct/n, 4)} \n Weighted F1: {round(weighted_f1, 4)} \n Unweighted F1: {round(unweighted_f1, 4)} \n Correct Remap: {good_remap} \n Total Remap: {total_remap}")
    if confusion_matrix:
        base_path = str(Path(file_name).parent)
        file_name = str(Path(file_name).name)
        print("Saving confusion matrix ...")
        write_confusion_matrix(base_path, file_name, logits, labels, class_names)
        print("Done")
def missing_entries(f1, f2):
    with open(f1, "r") as file1:
      d1 = json.load(file1)    
    with open(f2, "r") as file2:
      d2 = json.load(file2)
    paths1 = set([v["file+idx"] for _, v in d1.items()])
    paths2 = set([v["file+idx"] for _, v in d2.items()])
    return paths1 - paths2