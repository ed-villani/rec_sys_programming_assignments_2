import json

import numpy as np

from data_readers.content import Content
from data_readers.ratings import Ratings
from model.dict.content_dict import ContentDict
from model.dict.item_dict import ItemDict
from model.dict.user_dict import UserDict


class DictBuilder:
    def __new__(cls, ratings: Ratings, content: Content):
        item_dict = ItemDict()
        user_dict = UserDict()

        global_avg: float = 0.0

        for row in np.array(ratings.data):
            user_id = row[3]
            item_id = row[4]
            rating = row[1]

            item_dict.append(item_id)
            user_dict.append(user_id)

            item_dict[item_id].add_rating(rating, user_id)
            user_dict[user_id].add_rating(rating, item_id)

            global_avg = global_avg + rating

        global_avg = global_avg / len(ratings)

        content_dict = ContentDict()

        global_imdb: float = 0.0
        i_imdb: int = 0
        for row in np.array(content.data):
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
        return item_dict, user_dict, content_dict, global_avg, global_imdb
