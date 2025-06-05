import bcrypt


def hash_password(password: str) -> bytes:
    salt = bcrypt.gensalt()
    pwd_bytes: bytes = password.encode()
    return bcrypt.hashpw(pwd_bytes, salt)


def validate_password(password: str, hashed_password: str) -> bool:
    # Преобразуем хэш пароля из строки в байты, если он хранится в виде строки
    if isinstance(hashed_password, str):
        hashed_password = hashed_password.encode('utf-8')  # Преобразуем строку в байты

    # Преобразуем пароль в байты и сравниваем с хэшированным паролем
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)
