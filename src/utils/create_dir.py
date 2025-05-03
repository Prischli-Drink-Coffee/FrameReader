import os


def create_directories_if_not_exist(directories: list):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
