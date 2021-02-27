from model.dict.base_dict import BaseDict
from model.entity.item import Item


class ItemDict(BaseDict):
    def __init__(self):
        super().__init__()

    def __getitem__(self, y: str) -> Item:
        """ x.__getitem__(y) <==> x[y] """
        return self._data[y]

    def append(self, key: str) -> None:
        super(ItemDict, self).append(key, Item)
