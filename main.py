import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from model.attribute_matrix import AttributeMatrix
from model.dict.content_dict import ContentDict
from model.dict.item_dict import ItemDict
from model.dict.user_dict import UserDict


def read_input_and_split_tuples(file_path: str) -> pd.DataFrame:
    """

    Read the data and splits its column 'UserId:ItemId' in two column

    :param file_path: path to ratings file
    """
    data = pd.read_csv(file_path)
    data_split = data['UserId:ItemId'].str.split(":", n=1, expand=True)
    data['UserID'] = data_split[0]
    data['ItemID'] = data_split[1]
    # data.drop(columns=["UserId:ItemId"], inplace=True)
    return data


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


def main():
    item_dict = ItemDict()
    user_dict = UserDict()

    ratings = read_input_and_split_tuples('inputs/ratings.csv')
    global_avg: float = 0.0

    for row in tqdm(np.array(ratings)):
        user_id = row[3]
        item_id = row[4]
        rating = row[1]

        item_dict.append(item_id)
        user_dict.append(user_id)

        item_dict[item_id].add_rating(rating, user_id)
        user_dict[user_id].add_rating(rating, item_id)

        global_avg = global_avg + rating

    global_avg = global_avg / len(ratings)

    content = pd.read_csv('inputs/content.csv', sep='\t')
    content_split = content['ItemId,Content'].str.split(",", n=1, expand=True)
    content['Content'] = content_split[1]
    content['ItemID'] = content_split[0]

    content_dict = ContentDict()

    global_imdb: float = 0.0
    i_imdb: int = 0
    for row in tqdm(np.array(content)):
        item_id = row[2]
        item_content = json.loads(row[1])

        if not eval(item_content['Response']):
            continue

        item_dict.append(item_id)
        item_dict[item_id].add_content(item_content, content_dict)

        imdb_rate = item_dict[item_id].imdb_rate
        if isinstance(imdb_rate, float):
            global_imdb += imdb_rate
            i_imdb += 1

    global_imdb = global_imdb / i_imdb

    sm = cosine_similarity(AttributeMatrix(item_dict, content_dict))

    targets = read_input_and_split_tuples('inputs/targets.csv')

    solution: np.darray = np.zeros(len(targets))
    print("Predicting Rates")
    for index, data in tqdm(enumerate(np.array(targets))):
        user_id = data[1]
        item_id = data[2]

        if user_id in user_dict and item_id in item_dict:
            user = user_dict[user_id]
            item = item_dict[item_id]
            user_rates = user.rates

            all_sm = sm[item.alias_id]
            user_item_similarities = all_sm[[item_dict[item_user].alias_id for item_user in user_rates]]

            user_rates_values = np.array([value for value in user_rates.values()])

            solution[index] = round(np.average(user_rates_values, weights=user_item_similarities), 4)

        elif user_id in user_dict and item_id not in item_dict:
            user = user_dict[user_id]
            solution[index] = round(user.average_rate, 4)

        elif user_id not in user_dict and item_id in item_dict:
            item = item_dict[item_id]
            try:
                imdb_ratings = item.imdb_rate
                item_avg = item.average_rate
                if imdb_ratings != 0:
                    solution[index] = round(imdb_ratings, 4)
                elif item_avg is not None:
                    solution[index] = round(item_avg, 4)
                else:
                    solution[index] = round(global_imdb, 4)

            except KeyError:
                solution[index] = round(item.average_rate, 4)
        else:
            solution[index] = round(global_imdb, 4)

    print("Generating output file")
    targets["Prediction"] = solution
    targets.drop(columns=["UserID", "ItemID"], inplace=True)
    targets.to_csv('output.csv', index=False)


if __name__ == '__main__':
    main()
