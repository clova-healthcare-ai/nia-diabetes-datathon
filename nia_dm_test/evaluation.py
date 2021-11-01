import argparse
import os

import numpy as np
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')


def read_prediction(prediction_file):
    # NEED TO IMPLEMENT #1
    # function that loads prediction
    pred_array = np.loadtxt(prediction_file, dtype=np.int16)
    return pred_array


def read_ground_truth(ground_truth_file):
    # NEED TO IMPLEMENT #2
    # function that loads test_data
    gt_array = np.loadtxt(ground_truth_file, dtype=np.int16)
    return gt_array


# roc_auc_score
def evaluate(y_true, y_pred):
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        roc_auc = 0.5
        warnings.warn("Erroneous AUROC as only one class exists")
        pass

    return roc_auc


# user-defined function for evaluation metrics
def evaluation_metrics(ground_truth_file, prediction_file):
    # read prediction and ground truth from file
    gts = read_ground_truth(ground_truth_file)
    preds = read_prediction(prediction_file) # NOTE: prediction is text
    return evaluate(gts, preds)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # --prediction is set by file's name that contains the result of inference. (nsml internally sets)
    # prediction file requires type casting because '\n' character can be contained.
    args.add_argument('--prediction', type=str, default='pred.txt')
    args.add_argument('--test_label_path', type=str)
    config = args.parse_args()

    # test_label_path = os.path.join(DATASET_PATH, 'test', 'test_label')
    # print the evaluation result
    # evaluation prints only int or float value.
    print(evaluation_metrics(config.test_label_path, config.prediction))