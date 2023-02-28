"""Calculates the measures for the PAN19 hyperpartisan news detection task"""
# Modifications include: argparse, warning, division by zero handling

import argparse
import json
import os
import sys
import warnings
from collections import Counter
from xml.etree.ElementTree import iterparse

import numpy as np
import matplotlib.pyplot as plt
import itertools


"""
Generates a confusion matrix for the given data
"""
def plot_confusion_matrix(cm, labels, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], "d"),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", size=36)

    plt.tight_layout()
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def main(args):
    groundTruth = {}

    for _, elem in iterparse(args.inputDataset):
        if elem.tag != 'article': continue
        groundTruth[elem.get('id')] = elem.get('hyperpartisan')
        elem.clear()

    c = Counter()

    for line in args.inputRun:
        values = line.rstrip('\n').split()
        articleId, prediction = values[:2]

        c[(prediction, groundTruth[articleId])] += 1

    if sum(c.values()) < len(groundTruth):
        warnings.warn("Missing {} predictions".format(len(groundTruth) - sum(c.values())), UserWarning)

    tp = c[('true', 'true')]
    tn = c[('false', 'false')]
    fp = c[('true', 'false')]
    fn = c[('false', 'true')]

    # Code for generating confusion matrices
    cm = np.array([[tp, fn], [fp, tn]])
    plot_confusion_matrix(cm, ("hyperpartisan", "non-hyperpartisan"), "XGBoost")

    accuracy  = (tp + tn) / sum(c.values())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    results = {"truePositives": tp, "trueNegatives": tn, "falsePositives": fp, "falseNegatives": fn,
               "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    json.dump(results, args.outputFile, indent=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--inputDataset", type=argparse.FileType('rb'), required=True)
    parser.add_argument("-r", "--inputRun", type=argparse.FileType('r'), required=True)
    parser.add_argument("-o", "--outputFile", type=argparse.FileType('w'), default=sys.stdout)

    args=parser.parse_args()

    main(args)

    args.outputFile.close()
