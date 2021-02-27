import numpy as np

from data_readers.rec_sys_data import RecSysData
from model.dict.item_dict import ItemDict
from model.dict.user_dict import UserDict


class Targets(RecSysData):
    def __init__(self, file_path: str):
        super(Targets, self).__init__(file_path)
        """

        Initialize an dataset from its path and save its solution 

        :param file_path: path to ratings file 
        """
        self._solution: np.ndarray = None

    @property
    def solution(self):
        """

        Get the current solution for an item-based rec sys

        :return: the current solution
        """
        return self._solution

    def to_csv(self, out_path: str):
        """

        If the current solution is not None, save the solution to the desirable output file
        and put it in the correct standard for Kaggle

        :param out_path: path to save the file
        """
        if self._solution is not None:
            print("Generating output file")
            self.data["Prediction"] = self._solution
            self.data.drop(columns=["UserID", "ItemID"], inplace=True)
            self.data.to_csv(out_path, index=False)

    def solve(self, item_dict: ItemDict, user_dict: UserDict, global_imdb: float, sm: np.ndarray):
        """

        Predict the ratings by using a item-based model. Set user avg if item not
        in ItemDict, item avg if user not in UserDict, if none of them is in any
        dict, rate is the avg then.

        :param item_dict: Dict of item contains useful data to calculate de solution
        :param user_dict: Dict of user contains useful data to calculate de solution
        :param avg: global avg of all items ratings
        """
        self._solution: np.darray = np.zeros(len(self.data))
        print("Predicting Rates")
        for index, data in enumerate(np.array(self.data)):
            user_id = data[1]
            item_id = data[2]

            if user_id in user_dict and item_id in item_dict:
                user = user_dict[user_id]
                item = item_dict[item_id]
                user_rates = user.rates

                all_sm = sm[item.alias_id]
                user_item_similarities = all_sm[[item_dict[item_user].alias_id for item_user in user_rates]]

                user_rates_values = np.array([value for value in user_rates.values()])

                self._solution[index] = round(np.average(user_rates_values, weights=user_item_similarities), 4)

            elif user_id in user_dict and item_id not in item_dict:
                user = user_dict[user_id]
                self._solution[index] = round(user.average_rate, 4)

            elif user_id not in user_dict and item_id in item_dict:
                item = item_dict[item_id]
                try:
                    imdb_ratings = item.imdb_rate
                    item_avg = item.average_rate
                    if imdb_ratings != 0:
                        self._solution[index] = round(imdb_ratings, 4)
                    elif item_avg is not None:
                        self._solution[index] = round(item_avg, 4)
                    else:
                        self._solution[index] = round(global_imdb, 4)

                except KeyError:
                    self._solution[index] = round(item.average_rate, 4)
            else:
                self._solution[index] = round(global_imdb, 4)
