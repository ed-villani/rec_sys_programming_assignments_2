from model.dict.base_dict import BaseDict


class ContentDict(BaseDict):
    def __init__(self):
        super().__init__()
        self._data = {
            "Win": 0,
            "Nomination": 1,
            "BAFTA Won": 2,
            "BAFTA Nomination": 3,
            "Oscar Won": 4,
            "Oscar Nomination": 5,
            "Golden Won": 6,
            "Golden Nomination": 7,
            "Primetime Emmy Nomination": 8,
            "Primetime Emmy Won": 9,
            "Year": 10,
            "Runtime": 11,
            "imdbRating": 12
        }

    def append(self, key) -> None:
        if key not in self._data:
            self._data[key] = len(self._data.keys())
