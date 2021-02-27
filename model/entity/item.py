from model.entity.base_entity import Entity


class Item(Entity):
    def __init__(self, item_id: str, alias_id: int):
        super().__init__(item_id, alias_id)
