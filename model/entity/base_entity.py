class Entity():
    def __init__(self, entity: str, alias_id: int):
        self._data: dict = {
            "id": entity,
            "alias_id": alias_id,
            "rates": {
                "average": 0.0,
                "rates": {}
            }
        }

    @property
    def rates(self) -> dict:
        return self._data['rates']['rates']

    @property
    def average_rate(self) -> float:
        return self._data['rates']['average']

    def __str__(self) -> str:
        return str(self._data)

    __repr__ = __str__

    def add_rating(self, rating: int, item_id: str) -> None:
        self._data['rates']['rates'][item_id] = rating
        number_ratings = len(self._data['rates']['rates'].keys())
        if number_ratings == 1:
            self._data['rates']['average'] = rating
        else:
            value = (self._data['rates']['average'] * (number_ratings - 1) + rating) / number_ratings
            self._data['rates']['average'] = value

    def __getitem__(self, y: str):
        """ x.__getitem__(y) <==> x[y] """
        return self._data[y]

    def __setitem__(self, y, value):
        self._data[y] = value

    def __call__(self):
        return self._data

    def __contains__(self, key) -> bool:
        return key in self._data
