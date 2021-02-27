from data_readers.rec_sys_data import RecSysData


class Content(RecSysData):
    def __init__(self, file_path: str):
        """

        Initialize the content dataset from its path. n selects how many ratings an item
        must have to not be filtered from the list

        :param file_path: path to ratings file
        """
        super(Content, self).__init__(file_path)
