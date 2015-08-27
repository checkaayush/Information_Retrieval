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
        sum_vector1 = 0.0
        sum_vector1 = sum([(vector1[i] * vector1[i])
                          for i in range(len_vector1)])
        norm_vector1 = math.sqrt(sum_vector1)

        # Normalize the second vector
        sum_vector2 = 0.0
        sum_vector2 = sum([(vector2[i] * vector2[i])
                          for i in range(len_vector2)])
        norm_vector2 = math.sqrt(sum_vector2)

        # Calculate cosine simmilarity
        cos_sim = (dot / (norm_vector1 * norm_vector2))
        return cos_sim

    else:
        print "\nDimensions of input vectors must match.\n"
        return None
