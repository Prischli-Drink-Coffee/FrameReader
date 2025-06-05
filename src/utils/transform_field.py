from typing import Dict
import inspect


def transform_field(field: str,
                    iter_data: Dict,
                    get_entity_by_id):

    name_entity = field.split("_id")[0]
    entity_id = iter_data.get(field)

    if entity_id:
        # Проверяем, является ли get_entity_by_id функцией или объектом
        if callable(get_entity_by_id):
            # Инспектируем сигнатуру функции для проверки наличия аргумента dirs
            sig = inspect.signature(get_entity_by_id)
            if 'dirs' in sig.parameters:
                entity = get_entity_by_id(entity_id, dirs=True)
            else:
                entity = get_entity_by_id(entity_id)
        else:
            raise ValueError("Argument get_entity_by_id must be a function")

        try:
            iter_data[name_entity] = entity.model_dump(by_alias=True)
        except AttributeError:
            iter_data[name_entity] = entity
        del iter_data[field]

    return iter_data

