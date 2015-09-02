"""Implementation of various semantic similarity measures."""

import math


def cosine_similarity(vector1, vector2):
    """Calculates Cosine similarity between the two input vectors.

    Args:
        vector1 (list):
        vector2 (list):
                        Vectors as lists of weights of type 'float'.

    Returns:
        float: Value of cosine similarity (None if calculation not possible.)
    """

    len_vector1 = len(vector1)
    len_vector2 = len(vector2)

    if len_vector1 == len_vector2:
        dot = sum([(vector1[i] * vector2[i]) for i in range(len_vector1)])

        # Normalize the first vector
        sum_vector1 = sum([(vector1[i] * vector1[i])
                          for i in range(len_vector1)])
        norm_vector1 = math.sqrt(sum_vector1)

        # Normalize the second vector
        sum_vector2 = sum([(vector2[i] * vector2[i])
                          for i in range(len_vector2)])
        norm_vector2 = math.sqrt(sum_vector2)

        # Calculate cosine simmilarity
        denominator = (norm_vector1 * norm_vector2)
        if denominator:
            cos_sim = (float(dot) / denominator)
            return cos_sim
        else:
            "Zero division error"
            return 0.0
    else:
        print "\nDimensions of input vectors must match.\n"
        return 0.0


def euclidean_distance(vector1, vector2):
    """Calculates Euclidean Distance between the two input vectors.

    Args:
        vector1 (list):
        vector2 (list):
                        Vectors as lists of term frequencies.
    Returns:
        float: Value of Euclidean Distance.
    """

    length = len(vector1)
    addition = 0.0
    for i in range(length):
        value = (vector1[i] - vector2[i]) ** 2
        addition += value

    e_dist = math.sqrt(addition)
    return e_dist
