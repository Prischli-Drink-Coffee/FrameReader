from typing import Callable, List, Any
from fastapi import HTTPException, status
from src.utils.custom_logging import setup_logging

log = setup_logging()


def check_for_duplicates(
    get_all: Callable[[], List[Any]],
    check_id: int,
    attr_name: str,
    attr_value: Any,
    exception_detail: str
):
    existing_items = get_all()
    for item in existing_items:
        if int(item.ID) == int(check_id):
            continue
        if getattr(item, attr_name) == attr_value:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=exception_detail)


def check_if_exists(
    get_all: Callable[[], List[Any]],
    attr_name: str,
    attr_value: Any,
    exception_detail: str,
):
    existing_items = get_all()
    for item in existing_items:
        if getattr(item, attr_name) == attr_value:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=exception_detail
            )


def return_id_if_exists(
    get_all: Callable[[], List[Any]],
    attr_name: str,
    attr_value: Any
):
    existing_items = get_all()
    for item in existing_items:
        if getattr(item, attr_name) == attr_value:
            return item.ID
    return None
