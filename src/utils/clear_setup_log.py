import os
from src import path_to_project


def clear_setup_log():
    # Собираем полный путь к файлу setup.log
    log_file_path = os.path.join(path_to_project(), "setup.log")

    try:
        # Проверяем существование файла
        if os.path.isfile(log_file_path):
            # Открываем файл в режиме записи и очищаем его
            # with open(log_file_path, 'w') as log_file:
            #     log_file.truncate(0)
            os.remove(log_file_path)
        else:
            print(f"Файл setup.log не существует в указанной директории {log_file_path}.")
    except Exception as e:
        print(f"Произошла ошибка при очистке файла setup.log: {e}")


if __name__ == "__main__":
    clear_setup_log()
