import os
import pymysql
from src.utils.custom_logging import get_logger
from load_dotenv import load_dotenv


load_dotenv()
log = get_logger(__name__)


class CreateSQL:

    def __init__(self):
        self.path_to_sql = os.path.join(os.path.dirname(os.path.dirname(__file__)), f"{os.getenv('DB')}.sql")

        self.connection = pymysql.connect(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT")),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=30
        )

    def read_sql(self):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{os.getenv('DB')}`")
                cursor.execute(f"USE `{os.getenv('DB')}`")

                with open(self.path_to_sql, "r", encoding="utf-8") as f:
                    sql_script = f.read()

                    statements = [stmt.strip() for stmt in sql_script.split(';') if stmt.strip()]

                    for statement in statements:
                        try:
                            cursor.execute(statement)
                            log.info("Executed SQL statement: %s", statement)
                        except pymysql.MySQLError as e:
                            log.warning("SQL Warning: %s")

                self.connection.commit()
                log.info("Database was created and SQL script executed successfully")
        except Exception as ex:
            log.warning("Error during SQL script execution", exc_info=ex)
        finally:
            self.connection.close()


if __name__ == "__main__":
    create_sql = CreateSQL()
    create_sql.read_sql()
