import os
import pymysql
from pymysql.err import OperationalError
from src.utils.custom_logging import get_logger
from load_dotenv import load_dotenv

load_dotenv()
log = get_logger(__name__)


class Database:
    def __init__(self):
        self.connection = pymysql.connect(
            host=os.getenv("DB_HOST"),
            db=os.getenv("DB"),
            port=int(os.getenv("DB_PORT")),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
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
