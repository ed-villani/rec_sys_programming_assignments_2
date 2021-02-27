from model.entity.user import User


class BaseDict:
    def __init__(self):
        self._data = {}
        self._number_itens = 0

    def append(self, key: str, cls) -> None:
        if key not in self._data:
            self._data[key] = cls(key, self._number_itens)
            self._number_itens = self._number_itens + 1

    def __str__(self) -> str:
        return str(self._data)

    __repr__ = __str__

    def __getitem__(self, y: str) -> User:
        """ x.__getitem__(y) <==> x[y] """
        return self._data[y]

    def __contains__(self, key) -> bool:
        return key in self._data

    def __len__(self):
        return len(self._data)

    def __call__(self):
        return self._data

    def __iter__(self):
        return iter(self._data)

