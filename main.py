import sys
import time

import numpy as np

from data_readers.content import Content
from data_readers.ratings import Ratings
from data_readers.targets import Targets
from model.attribute_matrix import AttributeMatrix
from model.dict_builder import DictBuilder


def cosine_similarity(matrix):
    """

            Generate a cosine similarity matrix

            :return: similarity matrix item v. item
            """
    print("Calculating Similarities")
    norm = (matrix * matrix).sum(0, keepdims=True) ** 0.5
    norm_arr = np.divide(matrix, norm, where=norm != 0)
    similarity_matrix = norm_arr.T @ norm_arr
    return similarity_matrix


def main(argv):
    print("Starting")
    start = time.time()

    ratings = Ratings(argv[1])
    content = Content(argv[0])
    targets = Targets(argv[2])

    item_dict, user_dict, content_dict, global_avg, global_imdb = DictBuilder(ratings, content)
    sm = cosine_similarity(AttributeMatrix(item_dict, content_dict))

    targets.solve(
        item_dict=item_dict,
        user_dict=user_dict,
        global_imdb=global_imdb,
        sm=sm
    )

    targets.to_csv('output.csv')
    end = time.time()
    print(f"Total Time: {round(end - start, 3)}s")


if __name__ == "__main__":
    main(sys.argv[1:])
