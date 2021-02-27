from data_readers.rec_sys_data import RecSysData


class Ratings(RecSysData):
    def __init__(self, file_path: str):
        """

        Initialize the ratings dataset from its path. n selects how many ratings an item
        must have to not be filtered from the list

        :param file_path: path to ratings file
        """
        super().__init__(file_path)

