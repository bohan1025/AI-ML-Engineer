import numpy as np
from scipy.spatial import distance


def oneToOneMatch(predicted_points_array, actual_points_array, min_dist):
    """Finds the one-to-one matches between two lists of points."""

    # make array of form [[x1 x2 x3 ...], [y1 y2 y3 ...]] to form [[x1 y1], [x2 y2], [x3 y3], ...]
    format_predicted = [[x, y] for x, y in zip(*predicted_points_array)]
    format_actual = [[x, y] for x, y in zip(*actual_points_array)]

    # sample data
    pred_array = np.array(format_predicted)  # not modified
    actual_array = np.array(format_actual)  # not modified
    pred_array_copy = np.array(format_predicted)  # modified
    actual_array_copy = np.array(format_actual)  # modified

    # store the original indices of each point in the input lists
    list1_indices = np.arange(pred_array_copy.shape[0])
    list2_indices = np.arange(actual_array_copy.shape[0])

    result = []  # initialize the result list

    while (
        list1_indices.size and list2_indices.size
    ):  # loop until one of the lists is empty
        # calculate pairwise distances
        dist_matrix = distance.cdist(pred_array_copy, actual_array_copy)
        if (
            np.min(dist_matrix) > min_dist
        ):  # threshold over which we don't think it's a match
            break
        min_idx = np.unravel_index(
            np.argmin(dist_matrix), dist_matrix.shape
        )  # find the closest pair as (indx_of_pair_in_list1, indx_of_pair_in_list2)

        # add the closest pair indices to the result
        result.append((list1_indices[min_idx[0]], list2_indices[min_idx[1]]))
        # and remove them from the list
        list1_indices = np.delete(list1_indices, min_idx[0])
        pred_array_copy = np.delete(pred_array_copy, min_idx[0], axis=0)
        list2_indices = np.delete(list2_indices, min_idx[1])
        actual_array_copy = np.delete(actual_array_copy, min_idx[1], axis=0)

    # print("Number of matches found: ", len(result))
    # print("Number of points in list 1: ", len(format_predicted))
    # print("Number of points in list 2: ", len(format_actual))
    # print(
    #     "Number of distinct points: ",
    #     len(format_predicted) + len(format_actual) - len(result),
    # )

    return result, pred_array, actual_array
