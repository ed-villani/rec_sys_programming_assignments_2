from model.dict.base_dict import BaseDict
from model.entity.user import User


class UserDict(BaseDict):
    def __init__(self):
        super().__init__()

    def __getitem__(self, y: str) -> User:
        """ x.__getitem__(y) <==> x[y] """
        return self._data[y]

    def append(self, key: str) -> None:
        super(UserDict, self).append(key, User)