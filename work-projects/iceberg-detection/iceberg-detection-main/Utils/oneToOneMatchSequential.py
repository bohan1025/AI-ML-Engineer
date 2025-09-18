import numpy as np
from Utils.oneToOneMatchRefined import oneToOneMatchRefined
from scipy.spatial import distance


def calculate_distance(coord1, coord2):
    # Calculate the Euclidean distance between two coordinates
    return np.linalg.norm(np.array(coord1) - np.array(coord2))


def oneToOneMatchSequential(arrays, max_dist):
    """Finds the one-to-one matches between multiple lists of coordinates. 2 points are
    considered a match if the distance between them is less than max_dist.
    Returns a dictionary with the coordinates as keys and the number of times
    they were found as values. The number of times a coordinate was found is
    equal to the number of times it was a match with another coordinate.
    This function is meant to be used when there are many lists of coordinates
    and we want to find the number of times each coordinate was found in any of
    the lists. For example, if we have 3 lists of coordinates and a coordinate
    is found in all 3 lists, it will appear in the output with a value of 3.
    If it is found in 2 lists, it will appear with a value of 2, etc.
    The output dictionary can be used to find the number of unique coordinates
    that were found in any of the lists, and the number of duplicates.

    Args:
        arrays (list): list of arrays of coordinates. Each array is of the form
            [[x1 x2 x3 ...], [y1 y2 y3 ...]]
        max_dist (float): maximum distance between two points to consider them a match

    output is of the form {(x1, y1): 1, (x2, y2): 2, ...}
    """
    # make array of form [[x1 x2 x3 ...], [y1 y2 y3 ...]] to form [[x1 y1], [x2 y2], [x3 y3], ...]
    formatted_arrays = [np.array([[x, y] for x, y in zip(*array)]) for array in arrays]
    # check the nb of duplicates to avoid confusion later, it's normal to have a smaller
    # output than input size if there are duplicates
    nb_duplicates = sum(
        [
            len(array) - len(set([tuple(coord) for coord in array]))
            for array in formatted_arrays
        ]
    )

    result = {}  # initialize the result dict

    # loop until there is only one array left. Match the first two arrays, then
    # match the result with the next array, etc.
    while len(formatted_arrays) > 1:
        (
            _,
            matches_coordinates,
            _,
            array_1_no_match,
            _,
            array_2_no_match,
        ) = oneToOneMatchRefined(formatted_arrays.pop(0), formatted_arrays[0], max_dist)

        # check size of matches_coordinates to avoid error dimension if it's empty
        if len(matches_coordinates) == 0:
            new_array = np.concatenate([array_1_no_match, array_2_no_match])
        else:
            new_array = np.concatenate(
                [matches_coordinates, array_1_no_match, array_2_no_match]
            )

        # update the result dict
        for coord in new_array:
            if tuple(coord) not in result.keys():
                result[tuple(coord)] = 1
        for coord in matches_coordinates:
            result[tuple(coord)] += 1

        formatted_arrays[0] = new_array

    # print("original size ", sum([len(array.T) for array in arrays]))
    # print("nb duplicates in original data ", nb_duplicates)
    # print("output size ", sum(result.values()))

    # output is of the form {(x1, y1): 1, (x2, y2): 2, ...}
    return result


# example
import random

if __name__ == "__main__":
    # Example usage:
    max_distance = 0.5
    n_arrays = 10
    n = 10
    arrays = [
        np.array(
            [
                [random.randrange(n) for _ in range(n)],
                [random.randrange(n) for _ in range(n)],
            ]
        )
        for _ in range(n_arrays)
    ]
    result = oneToOneMatchSequential(arrays, max_distance)
    print(result)
