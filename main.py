import json
import re
from copy import deepcopy

import pandas as pd
import numpy as np
from tqdm import tqdm


def awards_structure(awards):
    def extract_values(regex, string):
        value = re.findall(regex, string)
        if not len(value):
            return 0
        return sum([int(s_find) for s_find in re.findall(r'\b\d+\b', value[0])])

    s = awards.lower()

    structure = {
        "Win": extract_values(r'\b\d+\b win', s),
        "Nomination": extract_values(r'\b\d+\b nomination', s),
        "BAFTA Won": extract_values(r'won \b\d+\b bafta', s),
        "BAFTA Nomination": extract_values(r'nominated for \b\d+\b bafta', s),
        "Oscar Won": extract_values(r'won \b\d+\b oscar', s),
        "Oscar Nomination": extract_values(r'nominated for \b\d+\b oscar', s),
        "Golden Won": extract_values(r'won \b\d+\b golden', s),
        "Golden Nomination": extract_values(r'nominated for \b\d+\b golden', s),
        "Primetime Emmy Nomination": extract_values(r'nominated for \b\d+\b primetime', s),
        "Primetime Emmy Won": extract_values(r'won \b\d+\b primetime', s)
    }

    # This lines asserts if all regex are right
    # assert sum([int(s_find) for s_find in re.findall(r'\b\d+\b', s)]) == sum([d for d in structure.values()])

    return structure


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
    item_dict = {}
    user_dict = {}

    ratings = read_input_and_split_tuples('inputs/ratings.csv')
    global_avg = 0

    for row in tqdm(np.array(ratings)):
        user_id = row[3]
        item_id = row[4]
        rating = row[1]

        if item_id not in item_dict:
            item_dict[item_id] = {}
            item_dict[item_id]['alias_id'] = len(item_dict.keys()) - 1
            item_dict[item_id]['rates'] = {}
            item_dict[item_id]['rates']['average'] = 0
            item_dict[item_id]['rates']['rates'] = {}

        if user_id not in user_dict:
            user_dict[user_id] = {}
            user_dict[user_id]['alias_id'] = len(item_dict.keys()) - 1
            user_dict[user_id]['rates'] = {}
            user_dict[user_id]['rates']['average'] = 0
            user_dict[user_id]['rates']['rates'] = {}

        item_dict[item_id]['rates']['rates'][user_id] = rating
        set_avg(item_dict, rating, item_id)
        user_dict[user_id]['rates']['rates'][item_id] = rating
        set_avg(user_dict, rating, user_id)

        global_avg = global_avg + rating

    global_avg = global_avg / len(ratings)

    content = pd.read_csv('inputs/content.csv', sep='\t')
    content_split = content['ItemId,Content'].str.split(",", n=1, expand=True)
    content['Content'] = content_split[1]
    content['ItemID'] = content_split[0]
    content_dict = {
        "Win": 0,
        "Nomination": 1,
        "BAFTA Won": 2,
        "BAFTA Nomination": 3,
        "Oscar Won": 4,
        "Oscar Nomination": 5,
        "Golden Won": 6,
        "Golden Nomination": 7,
        "Primetime Emmy Nomination": 8,
        "Primetime Emmy Won": 9
    }
    global_imdb = 0
    i_imdb = 0
    for row in tqdm(np.array(content)):
        item_id = row[2]
        item_content = json.loads(row[1])

        if not eval(item_content['Response']):
            continue

        # print(item_content['Actors'])

        if item_id not in item_dict:
            item_dict[item_id] = {}
            item_dict[item_id]['alias_id'] = len(item_dict.keys()) - 1
            item_dict[item_id]['rates'] = {}
            item_dict[item_id]['rates']['average'] = None
            item_dict[item_id]['rates']['rates'] = {}

        item_dict[item_id]['content'] = {}
        item_dict[item_id]['content']['Title'] = item_content['Title']

        if 'imdbRating' not in content_dict:
            content_dict['imdbRating'] = len(content_dict.keys())

        imdb_rating = item_content['imdbRating']
        if imdb_rating != 'N/A':
            item_dict[item_id]['content']['imdbRating'] = float(imdb_rating)
            global_imdb = global_imdb + float(imdb_rating)
            i_imdb = i_imdb + 1
        else:
            item_dict[item_id]['content']['imdbRating'] = 0

        # if 'Awards' not in content_dict:
        #     content_dict['Awards'] = len(content_dict.keys())
        item_dict[item_id]['content']['Awards'] = awards_structure(item_content['Awards'])

        item_dict[item_id]['content']['Year'] = int(item_content['Year']) if item_content['Year'] != 'N/A' else 0
        if 'Year' not in content_dict:
            content_dict['Year'] = len(content_dict.keys())

        item_runtime = item_content['Runtime']
        if item_runtime == 'N/A':
            item_runtime = 0
        elif ' h ' not in item_runtime:
            item_runtime = int(item_runtime.replace(" min", ""))
        else:
            data = item_runtime.replace(" h ", ",").replace(" min", "").split(",")
            item_runtime = int(data[0]) * 60 + int(data[1])
        item_dict[item_id]['content']['Runtime'] = item_runtime
        if 'Runtime' not in content_dict:
            content_dict['Runtime'] = len(content_dict.keys())

        # TODO filtras N/As
        item_dict[item_id]['content']['Genre'] = {}
        if item_content['Genre'] != 'N/A':
            for g in item_content['Genre'].replace(" ", "").lower().split(','):
                if g not in content_dict:
                    content_dict[g] = len(content_dict.keys())
                item_dict[item_id]['content']['Genre'][g] = g

        item_dict[item_id]['content']['Country'] = {}
        if item_content['Country'] != 'N/A':
            for c in item_content['Country'].replace(" ", "").lower().split(','):
                if c not in content_dict:
                    content_dict[c] = len(content_dict.keys())
                item_dict[item_id]['content']['Country'][c] = c

        item_dict[item_id]['content']['Language'] = {}
        if item_content['Language'] != 'N/A':
            for l in item_content['Language'].replace(" ", "").lower().split(','):
                if l not in content_dict:
                    content_dict[l] = len(content_dict.keys())
                item_dict[item_id]['content']['Language'][l] = l

        item_dict[item_id]['content']['Director'] = {}
        if item_content['Director'] != 'N/A':
            for d in item_content['Director'].replace(", ", ",").lower().split(','):
                if d not in content_dict:
                    content_dict[d] = len(content_dict.keys())
                item_dict[item_id]['content']['Director'][d] = d

    global_imdb = global_imdb / i_imdb
    similarity = np.zeros((len(content_dict), len(item_dict)), dtype=np.float32)
    for item_id in tqdm(item_dict):
        if 'content' not in item_dict[item_id]:
            continue
        similarity[content_dict['Runtime']][item_dict[item_id]['alias_id']] = item_dict[item_id]['content']['Runtime']
        # similarity[content_dict['Awards']][item_dict[item_id]['alias_id']] = item_dict[item_id]['content']['Awards']
        similarity[content_dict['imdbRating']][item_dict[item_id]['alias_id']] = item_dict[item_id]['content'][
            'imdbRating']
        similarity[content_dict['Year']][item_dict[item_id]['alias_id']] = item_dict[item_id]['content']['Year']

        for g in item_dict[item_id]['content']['Genre']:
            similarity[content_dict[g]][item_dict[item_id]['alias_id']] = 1

        for c in item_dict[item_id]['content']['Country']:
            similarity[content_dict[c]][item_dict[item_id]['alias_id']] = 1

        for l in item_dict[item_id]['content']['Language']:
            similarity[content_dict[l]][item_dict[item_id]['alias_id']] = 1

        for d in item_dict[item_id]['content']['Director']:
            similarity[content_dict[d]][item_dict[item_id]['alias_id']] = 1

        for a in item_dict[item_id]['content']['Awards']:
            similarity[content_dict[a]][item_dict[item_id]['alias_id']] = item_dict[item_id]['content']['Awards'][a]

    norm_type = NormalizationType.MAX_MIN2
    # TODO normalizar esse role aqui. Faze func'ão de normalização
    similarity[content_dict['Runtime']] = row_normalize(similarity[content_dict['Runtime']], norm_type)
    # similarity[content_dict['Awards']] = row_normalize(similarity[content_dict['Awards']], NormalizationType.MAX_MIN)
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
        if user == "u0003691" and item == "i1922545":
            i = 0
        if user in user_dict and item in item_dict:
            user_rates = user_dict[user]['rates']['rates']

            all_sm = sm[item_dict[item]['alias_id']]
            div, user_item_similarities = np.sum(np.abs(all_sm[
                                                            [item_dict[item_user]['alias_id'] for item_user in
                                                             user_rates]])), all_sm[
                                              [item_dict[item_user]['alias_id'] for item_user in user_rates]]

            user_rates_values = np.array([value for value in user_rates.values()])

            user_med = user_dict[user]['rates']['average'] if user_dict[user]['rates']['average'] is not None else \
                item_dict[item]['content']['imdbRating']

            if div > 0.0001:
                solution[index] = round(np.dot(user_item_similarities, user_rates_values) / div, 4)
            else:
                solution[index] = round(user_med, 4)

        elif user in user_dict and item not in item_dict:
            solution[index] = round(user_dict[user]['rates']['average'], 4)
        elif user not in user_dict and item in item_dict:
            try:
                imdb_ratings = item_dict[item]['content']['imdbRating']
                item_avg = item_dict[item]['rates']['average']
                if imdb_ratings != 0:
                    solution[index] = round(imdb_ratings, 4)
                elif item_avg is not None:
                    solution[index] = round(item_avg, 4)
                else:
                    solution[index] = round(global_imdb, 4)

            except KeyError:
                solution[index] = round(item_dict[item]['rates']['average'], 4)
        else:
            solution[index] = round(global_imdb, 4)

    print("Generating output file")
    targets["Prediction"] = solution
    targets.drop(columns=["UserID", "ItemID"], inplace=True)
    targets.to_csv('output.csv', index=False)


if __name__ == '__main__':
    main()
