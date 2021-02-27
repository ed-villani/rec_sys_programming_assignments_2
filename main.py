import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from model.dict.content_dict import ContentDict
from model.dict.item_dict import ItemDict
from model.dict.user_dict import UserDict


class NormalizationType:
    SUM = 'sum'
    MAX_MIN = 'max-min'
    MAX_MIN2 = 'max-min2'
    Z_NORM = 'z-norm'


def row_normalize(row, type='max-min'):
    if type == 'sum':
        return row / np.sum(row)
    if type == 'max-min':
        return (row - np.min(row)) / (np.max(row) - np.min(row))
    if type == 'z-norm':
        return (row - np.mean(row)) / np.std(row)
    if type == 'max-min2':
        new_min = np.min(row) - 1
        return (row - new_min) / (np.max(row) - new_min)


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


def set_avg(entity_dict, rating, key):
    number_ratings = len(entity_dict[key]['rates']['rates'].keys())
    if number_ratings == 1:
        entity_dict[key]['rates']['average'] = rating
    else:
        value = (entity_dict[key]['rates']['average'] * (number_ratings - 1) + rating) / number_ratings
        entity_dict[key]['rates']['average'] = value


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

    similarity = np.zeros((len(content_dict), len(item_dict)), dtype=np.float32)
    for item_id in tqdm(item_dict):
        item = item_dict[item_id]

        if 'Content' not in item:
            continue

        item_alias_id = item.alias_id

        similarity[content_dict['Runtime']][item_alias_id] = item.runtime
        similarity[content_dict['imdbRating']][item_alias_id] = item.imdb_rate
        similarity[content_dict['Year']][item_alias_id] = item.year

        for g in item_dict[item_id]['Content']['Genre']:
            similarity[content_dict[g]][item_alias_id] = 1

        for c in item_dict[item_id]['Content']['Country']:
            similarity[content_dict[c]][item_alias_id] = 1

        for l in item_dict[item_id]['Content']['Language']:
            similarity[content_dict[l]][item_alias_id] = 1

        for d in item_dict[item_id]['Content']['Director']:
            similarity[content_dict[d]][item_alias_id] = 1

        for a in item.awards:
            similarity[content_dict[a]][item_alias_id] = item.awards[a]

    norm_type = NormalizationType.MAX_MIN2

    similarity[content_dict['Runtime']] = row_normalize(similarity[content_dict['Runtime']], norm_type)
    similarity[content_dict['imdbRating']] = row_normalize(similarity[content_dict['imdbRating']], norm_type)
    similarity[content_dict['Year']] = row_normalize(similarity[content_dict['Year']], norm_type)
    for index, k in enumerate(content_dict):
        if index > 9:
            break
        similarity[content_dict[k]] = row_normalize(similarity[content_dict[k]], NormalizationType.MAX_MIN)

    sm = cosine_similarity(similarity)

    targets = read_input_and_split_tuples('inputs/targets.csv')

    solution: np.darray = np.zeros(len(targets))
    print("Predicting Rates")
    for index, data in tqdm(enumerate(np.array(targets))):
        user = data[1]
        item = data[2]

        if user in user_dict and item in item_dict:
            user_rates = user_dict[user].rates

            all_sm = sm[item_dict[item].alias_id]
            div, user_item_similarities = np.sum(np.abs(all_sm[
                                                            [item_dict[item_user].alias_id for item_user in
                                                             user_rates]])), all_sm[
                                              [item_dict[item_user].alias_id for item_user in user_rates]]

            user_rates_values = np.array([value for value in user_rates.values()])

            user_med = user_dict[user].average_rate if user_dict[user].average_rate is not None else \
                item_dict[item].imdb_rate

            if div > 0.0001:
                solution[index] = round(np.dot(user_item_similarities, user_rates_values) / div, 4)
            else:
                solution[index] = round(user_med, 4)

        elif user in user_dict and item not in item_dict:
            solution[index] = round(user_dict[user].average_rate, 4)
        elif user not in user_dict and item in item_dict:
            try:
                imdb_ratings = item_dict[item].imdb_rate
                item_avg = item_dict[item].average_rate
                if imdb_ratings != 0:
                    solution[index] = round(imdb_ratings, 4)
                elif item_avg is not None:
                    solution[index] = round(item_avg, 4)
                else:
                    solution[index] = round(global_imdb, 4)

            except KeyError:
                solution[index] = round(item_dict[item].average_rate, 4)
        else:
            solution[index] = round(global_imdb, 4)

    print("Generating output file")
    targets["Prediction"] = solution
    targets.drop(columns=["UserID", "ItemID"], inplace=True)
    targets.to_csv('output.csv', index=False)


if __name__ == '__main__':
    main()
