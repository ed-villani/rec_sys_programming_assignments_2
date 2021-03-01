import numpy as np

from model.dict.content_dict import ContentDict
from model.dict.item_dict import ItemDict


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


class AttributeMatrix:
    def __new__(cls, item_dict: ItemDict, content_dict: ContentDict) -> np.ndarray:
        attr_matrix = np.zeros(
            (len(content_dict), len(item_dict)), dtype=np.float32)
        for item_id in item_dict:
            item = item_dict[item_id]

            if 'Content' not in item:
                continue

            item_alias_id = item.alias_id

            attr_matrix[content_dict['Metascore']
                        ][item_alias_id] = item.metascore
            attr_matrix[content_dict['Runtime']][item_alias_id] = item.runtime
            attr_matrix[content_dict['imdbRating']
                        ][item_alias_id] = item.imdb_rate
            attr_matrix[content_dict['Year']][item_alias_id] = item.year

            for g in item.genres:
                attr_matrix[content_dict[g]][item_alias_id] = 1

            for c in item.countries:
                attr_matrix[content_dict[c]][item_alias_id] = 1

            for l in item.languages:
                attr_matrix[content_dict[l]][item_alias_id] = 1

            for d in item.directors:
                attr_matrix[content_dict[d]][item_alias_id] = 1

            # for ac in item.actors:
            #     attr_matrix[content_dict[ac]][item_alias_id] = 1

            for a in item.awards:
                attr_matrix[content_dict[a]][item_alias_id] = item.awards[a]

        attr_matrix[content_dict['Runtime']] = row_normalize(
            attr_matrix[content_dict['Runtime']], NormalizationType.MAX_MIN2)
        attr_matrix[content_dict['imdbRating']] = row_normalize(
            attr_matrix[content_dict['imdbRating']], NormalizationType.MAX_MIN)
        attr_matrix[content_dict['Metascore']] = row_normalize(
            attr_matrix[content_dict['Metascore']], NormalizationType.MAX_MIN)
        attr_matrix[content_dict['Year']] = row_normalize(
            attr_matrix[content_dict['Year']], NormalizationType.MAX_MIN2)
        # for index, k in enumerate(content_dict):
        #     if index > 9:
        #         break
        #     attr_matrix[content_dict[k]] = row_normalize(attr_matrix[content_dict[k]], NormalizationType.MAX_MIN)
        return attr_matrix
