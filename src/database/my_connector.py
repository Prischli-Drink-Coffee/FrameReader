import pymysql
from pymysql.err import OperationalError
from src.utils.env import Env
from src.utils.custom_logging import setup_logging

env = Env()
log = setup_logging()


class Database:
    def __init__(self):
        self.connection = pymysql.connect(
            host=env.__getattr__("DB_HOST"),
            db=env.__getattr__("DB"),
            port=int(env.__getattr__("DB_PORT")),
            user=env.__getattr__("DB_USER"),
            password=env.__getattr__("DB_PASSWORD"),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )

    def check_and_reconnect(self):
        try:
            self.connection.close()
            self.connection.ping(reconnect=True)
        except OperationalError as e:
            log.exception(e)

    def execute_query(self, query, params=None):
        self.check_and_reconnect()
        with self.connection.cursor() as cursor:
            cursor.execute(query, params)
            self.connection.commit()
            return cursor

    def fetch_one(self, query, params=None):
        self.check_and_reconnect()
        with self.connection.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()

    def fetch_all(self, query, params=None):
        self.check_and_reconnect()
        with self.connection.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()


db = Database()
