import numpy as np
from scipy.spatial import distance


def oneToOneMatchRefined(array_1, array_2, max_dist):
    """Finds the one-to-one matches between two lists of coordinates. 2 points are
    considered a match if the distance between them is less than max_dist.
    Returns the matches, the points in the predicted array that were NOT matched,
    and the points in the actual array that were NOT matched."""

    # make array of form [[x1 x2 x3 ...], [y1 y2 y3 ...]] to form [[x1 y1], [x2 y2], [x3 y3], ...]
    # format_array_1 = np.array([(x, y) for x, y in zip(*array_1)])
    # format_array_2 = np.array([(x, y) for x, y in zip(*array_2)])
    format_array_1 = array_1
    format_array_2 = array_2
    # make a copy of the arrays to modify
    format_array_1_copy = np.copy(format_array_1)
    format_array_2_copy = np.copy(format_array_2)

    # store the original indices of each point in the input lists
    list1_indices = np.arange(format_array_1.shape[0])
    list2_indices = np.arange(format_array_2.shape[0])
    # make a copy of the indices to modify
    list1_indices_copy = np.copy(list1_indices)
    list2_indices_copy = np.copy(list2_indices)

    result = []  # initialize the result list

    while (
        list1_indices.size and list2_indices.size
    ):  # loop until one of the lists is empty
        # calculate pairwise distances
        dist_matrix = distance.cdist(format_array_1, format_array_2)
        if (
            np.min(dist_matrix) > max_dist
        ):  # threshold over which we don't think it's a match
            break
        min_idx = np.unravel_index(
            np.argmin(dist_matrix), dist_matrix.shape
        )  # find the closest pair as (indx_of_pair_in_list1, indx_of_pair_in_list2)

        # add the closest pair indices to the result
        result.append((list1_indices[min_idx[0]], list2_indices[min_idx[1]]))
        # and remove them from the list
        list1_indices = np.delete(list1_indices, min_idx[0])
        list2_indices = np.delete(list2_indices, min_idx[1])
        format_array_1 = np.delete(format_array_1, min_idx[0], axis=0)
        format_array_2 = np.delete(format_array_2, min_idx[1], axis=0)

    # print("Number of matches found: ", len(result))
    # print("Number of points in list 1: ", len(format_array_1_copy))
    # print("Number of points in list 2: ", len(format_array_2_copy))
    # print(
    #     "Number of distinct points: ",
    #     len(format_array_1_copy) + len(format_array_2_copy) - len(result),
    # )

    array_1_no_match = np.delete(format_array_1_copy, [x[0] for x in result], axis=0)
    array_2_no_match = np.delete(format_array_2_copy, [x[1] for x in result], axis=0)
    # For the coordinates of the matches, use array_1 (could use array_2 as well)
    matches_coordinates = np.array(
        [(format_array_1_copy[x[0]][0], format_array_1_copy[x[0]][1]) for x in result]
    )
    # Need to output the indices of the matches in the original arrays to be able to
    # use them for sets intersections
    array_1_no_match_ind = np.delete(list1_indices_copy, [x[0] for x in result], axis=0)
    array_2_no_match_ind = np.delete(list2_indices_copy, [x[1] for x in result], axis=0)
    matches_ind = np.array(result)

    return (
        matches_ind,
        matches_coordinates,
        array_1_no_match_ind,
        array_1_no_match,
        array_2_no_match_ind,
        array_2_no_match,
    )
