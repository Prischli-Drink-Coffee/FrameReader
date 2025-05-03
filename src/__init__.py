import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, project_path)


def path_to_config():
    return os.path.join(project_path, 'src/config.yaml')


def path_to_project():
    return os.path.join(project_path)


def path_to_logging():
    return os.path.join(project_path, 'src/logging.yaml')


def path_to_env():
    return os.path.join(project_path, '.env')

