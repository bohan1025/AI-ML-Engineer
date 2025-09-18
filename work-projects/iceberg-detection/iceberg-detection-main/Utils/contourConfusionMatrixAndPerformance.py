import numpy as np


def contourConfusionMatrixAndPerformance(result, pred_array, actual_array):
    confusion_matrix = np.zeros((2, 2))

    TP = len(result)
    FP = len(pred_array) - TP
    FN = len(actual_array) - TP
    TN = 0
    confusion_matrix[0, 0] = TP
    confusion_matrix[0, 1] = FP
    confusion_matrix[1, 0] = FN
    confusion_matrix[1, 1] = TN

    if TP + FP == 0:
        print("Warning: TP + FP = 0")
        precision = 0
    else:
        precision = TP / (TP + FP)
    if TP + FN == 0:
        print("Warning: TP + FN = 0")
        recall = 0
    else:
        recall = TP / (TP + FN)
    if precision + recall == 0:
        print("Warning: precision + recall = 0")
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    return confusion_matrix, precision, recall, f1_score
