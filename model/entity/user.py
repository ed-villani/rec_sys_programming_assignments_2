from model.entity.base_entity import Entity


class User(Entity):
    def __init__(self, user_id: str, alias_id: int):
        super().__init__(user_id, alias_id)
