def encode_list_to_string(numbers_list):
    """
    Функция принимает список чисел и возвращает строку,
    где числа разделены запятыми.

    Args:
        numbers_list (list): Список чисел.

    Returns:
        str: Строка, содержащая числа, разделенные запятыми.
    """
    # Преобразуем каждый элемент списка в строку и объединяем их через запятую
    return ','.join(map(str, numbers_list))


def decode_string_to_list(numbers_string):
    """
    Функция принимает строку и возвращает список чисел,
    проверяя строку на соответствие формату "число,число,...".

    Args:
        numbers_string (str): Строка, содержащая числа, разделенные запятыми.

    Returns:
        list: Список чисел.

    Raises:
        ValueError: Если строка не соответствует формату.
    """
    # Проверяем, что строка не пустая и состоит только из чисел и запятых
    if not numbers_string or not all(char.isdigit() or char == ',' for char in numbers_string):
        raise ValueError("Строка не соответствует формату 'число,число,...'")

    # Разделяем строку по запятым и преобразуем элементы обратно в числа
    return [int(num) for num in numbers_string.split(',')]

