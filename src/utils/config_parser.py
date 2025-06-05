import yaml
import codecs
from os import path


class ConfigParser:
    @staticmethod
    def parse(file: str):
        if not path.exists(file):
            raise Exception(f'File "{file}" not exists')
        try:
            with codecs.open(file, mode="r", encoding="latin1") as config_file:
                raw_data = config_file.read()
                decoded_data = raw_data.encode('latin1').decode('utf-8')
                config = yaml.load(decoded_data, Loader=yaml.SafeLoader)
        except Exception as config_parse_exception:
            raise Exception(f'Can not parse file "{file}"') from config_parse_exception
        return config
