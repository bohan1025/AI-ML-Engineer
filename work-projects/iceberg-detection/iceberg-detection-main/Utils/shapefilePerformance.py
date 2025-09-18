import numpy as np
from Utils.getPoints import pointsAsPixels
from Utils.contourConfusionMatrixAndPerformance import (
    contourConfusionMatrixAndPerformance,
)
from Utils.oneToOneMatch import oneToOneMatch


def shapefilePerformance(output_filename, image_name, gt, min_dist=100):
    # import relevant images and shp
    predicted_shp = f"Generated_Shapefiles/{output_filename}/{output_filename}.shp"
    actual_shp = f"Shapefiles/{image_name}/{image_name}.shp"

    # Open the SAR data and get transform
    px, py = pointsAsPixels(predicted_shp, gt)
    ax, ay = pointsAsPixels(actual_shp, gt)
    print("Number of predicted points: ", len(px))
    print("Number of actual points: ", len(ax))

    # min dist is in pixels
    result, pred_array, actual_array = oneToOneMatch(
        np.array([px, py]), np.array([ax, ay]), min_dist
    )
    result = contourConfusionMatrixAndPerformance(result, pred_array, actual_array)

    confusion_matrix, precision, recall, f1_score = result
    # Print the results, should add accuracy
    print("Confusion Matrix:\n", confusion_matrix)
    # print("Precision (high Precision, low FP):", precision)
    # print("Recall (high Recall, low FN):", recall)
    print("F1 Score:", f1_score)
    return result
