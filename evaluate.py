# -*- coding: utf-8 -*-
"""
Evaluate trained models with test-data.

Example Usage:
    - Plot a CSV ROC file: `python evaluate.py --plot_csv "/path/to/file.csv"`
    - Evaluate specfic model: `python evaluate.py --model_path "/path/to/model.h5"`
"""

from predict import predict_images_path
from factories import model_factory
from helpers import prowl, io
import keras
from keras import models
from keras.utils import to_categorical
from sklearn.metrics import roc_curve, precision_recall_curve, auc, balanced_accuracy_score
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import sys
import argparse
import sty
from sty import fg, bg, ef, rs



def write_to_csv(x, y, thresholds, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    length = min(len(x), len(y), len(thresholds))
    with open(filepath, mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['THRESHOLDS', 'X', 'Y'])

        for i in range(length):
            writer.writerow([thresholds[i], x[i], y[i]])


def plot_to_pdf(roc_auc, fpr, tpr, thresholds, title, labels, pdf_save_path):
    plt.figure()
    lns1 = plt.plot(fpr, tpr, label=labels[0], color='black', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    plt.title(title)

    # Add Thresholds to Plot
    ax2 = plt.gca().twinx()
    min_len = min(len(fpr), len(thresholds))
    lns2 = ax2.plot(fpr[:min_len], thresholds[:min_len], label=f'Classification Thresholds', markeredgecolor='r',linestyle='dashed', color='r')
    ax2.set_ylim([thresholds[-1],thresholds[0]])
    ax2.tick_params(axis='y', colors='red')
    # ax2.set_xlim([fpr[0],fpr[-1]])

    # Legend
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc="lower right")

    plt.savefig(pdf_save_path, bbox_inches='tight')


def roc_auc(y_true, y_predicted, save_dir=False, save_prefix=""):
    """
    Reference: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    """
    min_len = min(len(y_true), len(y_predicted))
    y_true, y_predicted = y_true[:min_len], y_predicted[:min_len]

    # Calculate ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_predicted)
    roc_auc = auc(fpr, tpr) 
    
    # Save Plot as PDF & CSV
    if save_dir:
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        pdf_save_path = os.path.join(save_dir, f'{save_prefix} ROC (AUC={roc_auc:.2f}).pdf')
        plot_labels = [f'ROC Curve (AUC = {roc_auc:0.2f})', '1 - Speciﬁcity (False-Positive-Rate)', 'Sensitivity (True-Positive-Rate)']
        plot_to_pdf(roc_auc, fpr, tpr, thresholds, "Receiver Operating Characteristic", plot_labels, pdf_save_path)

        csv_save_path = os.path.join(save_dir, f'{save_prefix} ROC (AUC={roc_auc:.2f}).csv')
        write_to_csv(fpr, tpr, thresholds, csv_save_path)

    return roc_auc, fpr, tpr, thresholds


def precision_recall_auc(y_true, y_predicted, save_dir=False, save_prefix=""):
    """
    Reference: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
    """
    min_len = min(len(y_true), len(y_predicted))
    y_true, y_predicted = y_true[:min_len], y_predicted[:min_len]

    # Calculate ROC
    precision, recall, thresholds = precision_recall_curve(y_true, y_predicted, 0)
    precision, recall, thresholds = np.append(precision, 1.0), np.append(recall, 0.0), np.append(thresholds, 1.0)
    precision_recall_auc = auc(recall, precision) 

    # Save Plot as PDF & CSV
    if save_dir:
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        pdf_save_path = os.path.join(save_dir, f'{save_prefix} PR (AUC={precision_recall_auc:.2f}).pdf')
        plot_labels = [f'Precision Recall Curve (AUC = {precision_recall_auc:0.2f})', 'Sensitivity (Recall)', 'Precision']
        plot_to_pdf(precision_recall_auc, recall, precision, thresholds, "Precision-Recall Curve", plot_labels, pdf_save_path)

        csv_save_path = os.path.join(save_dir, f'{save_prefix} PR (AUC={precision_recall_auc:.2f}).csv')
        write_to_csv(recall, precision, thresholds, csv_save_path)

    return precision_recall_auc, precision, recall, thresholds


def evaluate_model(model_path, model_type, test_images_path, image_size=250):
    print("\nEvaluate Model..")
    print(sty.ef.dim + f"[model_path={model_path}, test_images_path={test_images_path}, image_size={image_size}]\n" + sty.rs.all)

    files, _, single_value_predictions, class_predictions = predict_images_path(model_path, model_type, test_images_path, image_size=image_size)
    
    y_true = [os.path.basename(os.path.dirname(path)) for path in files]
    y_true = [0 if x.lower() == 'benign' else 1 for x in y_true]
    y_true = np.asarray(y_true)

    # Calculate AUCs and save results in Model-Directory
    model_dir, model_name = os.path.dirname(model_path), os.path.splitext(os.path.basename(model_path))[0]

    auc_1, _, _, _ = roc_auc(y_true, single_value_predictions, save_dir=model_dir, save_prefix=model_name)
    auc_2, _, _, _ = precision_recall_auc(y_true, single_value_predictions, save_dir=model_dir, save_prefix=model_name)

    return auc_1, auc_2
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model_path', type=str, nargs='?')
    group.add_argument('--latest_models', type=int, default=0)
    parser.add_argument('--models_dir', type=str, nargs='?', default='./results/models/')
    parser.add_argument('--keep_best_only', action='store_const', const=True)

    parser.add_argument('--test_images_path', type=str, nargs='?', default='./data/test')
    parser.add_argument('--model_type', type=int, required=True, choices=range(0, model_factory.count()))
    parser.add_argument('--image_size', type=int, nargs='?', default=250)

    parser.add_argument('-n', '--notify', action='store_const', const=True)
    args = parser.parse_args()

    if not os.path.isdir(args.test_images_path):
        raise ValueError(f"Directory '{args.test_images_path}' (test_images_path) doesn't exist")
    if args.latest_models and not os.path.isdir(args.models_dir):
        raise ValueError(f"Directory '{args.models_dir}' (models_dir) doesn't exist")

    model_paths = io.latest_files(args.models_dir, ('.h5', '.h5df'), args.latest_models) if args.latest_models else [args.model_path]

    if args.notify and len(model_paths) > 1:
        prowl.send_message(f'Start Evaluation of {len(model_paths)} Models..', '', priority=1)
    elif args.notify:
        prowl.send_message(f'Start Evaluation of Model..', '', priority=1)

    best_model_idx, highest_auc = 0, 0.
    for idx, model_path in enumerate(model_paths):
        auc_1, auc_2 = evaluate_model(model_path, args.model_type, args.test_images_path, args.image_size)

        # if result > highest_auc:
        #     highest_auc = result
        #     best_model_idx = idx

        print(bg.blue + f"\nROC-AUC = '{auc_1:.2f}', PR-AUC = '{auc_2:.2f}'" + bg.rs)
        if args.notify: prowl.send_message(f'Finished Evaluation {idx + 1}/{len(model_paths)} ✅', f"\nROC-AUC = '{auc_1:.2f}', PR-AUC = '{auc_2:.2f}'", priority=1)

    # If `args.keep_best_only` was set delete all except the best model-files
    # if len(model_paths) > 1 and args.keep_best_only:
    #     print(f"Keeping only best model '{model_paths[best_model_idx]}'")
    #     for idx, model_path in enumerate(model_paths):
    #         if idx != best_model_idx and os.path.exists(model_path):
    #             print(f"Deleting model '{model_path}'..")
    #             os.remove(model_path)
            
    
    
