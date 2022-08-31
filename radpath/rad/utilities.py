import csv
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from sklearn import metrics
import pandas as pd

def get_output_target(test_csv, result_file, label, label_dict):
    # groundtruth
    test_df = pd.read_csv(test_csv)
    gt = [label_dict[test_df.loc[idx, label]] for idx in range(len(test_df))]
    # prediction
    pred = []
    with open(result_file, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            prob = np.array(list(map(float, row)))
            pred.append(np.argmax(prob))
    return pred, gt

def metrics_all(test_csv, result_file, label, label_dict):
    labels = [0,1,2]
    pred, gt = get_output_target(test_csv, result_file, label, label_dict)
    recall = metrics.recall_score(gt, pred, average=None)
    precision = metrics.precision_score(gt, pred, average=None, zero_division=0)
    cm = metrics.confusion_matrix(gt, pred, labels=labels)
    f1 =  metrics.f1_score(gt, pred, labels = labels, average='micro')
    bal_acc = metrics.balanced_accuracy_score(gt, pred)
    kappa = metrics.cohen_kappa_score(gt, pred)
    return precision, recall, cm, f1, bal_acc, kappa #auroc

