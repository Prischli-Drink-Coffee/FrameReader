def encode_list_to_string(numbers_list):
    return ','.join(map(str, numbers_list))


def decode_string_to_list(numbers_string):
    if not numbers_string or not all(char.isdigit() or char == ',' for char in numbers_string):
        raise ValueError("Строка не соответствует формату 'число,число,...'")
    return [int(num) for num in numbers_string.split(',')]

