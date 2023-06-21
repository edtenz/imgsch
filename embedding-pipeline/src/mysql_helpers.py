import pymysql

from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PWD, MYSQL_DB, DEFAULT_TABLE
from logger import LOGGER


class MysqlClient(object):

    def __init__(self, host: str = MYSQL_HOST, port: int = MYSQL_PORT,
                 user: str = MYSQL_USER, pwd: str = MYSQL_PWD,
                 db: str = MYSQL_DB):
        self.conn = pymysql.connect(host=host, user=user, port=port, password=pwd,
                                    database=db,
                                    local_infile=True)
        self.cursor = self.conn.cursor()

    def test_connection(self):
        try:
            self.conn.ping()
        except Exception:
            self.conn = pymysql.connect(host=MYSQL_HOST, user=MYSQL_USER, port=MYSQL_PORT, password=MYSQL_PWD,
                                        database=MYSQL_DB, local_infile=True)
            self.cursor = self.conn.cursor()

    def create_table(self, table_name: str = DEFAULT_TABLE):
        # Create mysql table if not exists
        self.test_connection()
        sql = """
        CREATE TABLE IF NOT EXISTS {}(
          id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
          image_key VARCHAR(50) NOT NULL,
          box VARCHAR(30),
          score FLOAT,
          INDEX idx_image_key (image_key),
          UNIQUE KEY idx_image_box (image_key, box)
        );
        """.format(table_name)
        try:
            drop_sql = f"DROP TABLE IF EXISTS {table_name};"
            self.cursor.execute(drop_sql)
            LOGGER.debug(f"MYSQL delete table:{table_name}")
            self.cursor.execute(sql)
            LOGGER.debug(f"MYSQL create table: {table_name} with sql: {sql}")
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            raise Exception("MYSQL ERROR: {} with sql: {}".format(e, sql))

    def insert_into(self, image_key: str, box: str, score: float = 0.0, table_name: str = DEFAULT_TABLE) -> bool:
        # Batch insert (Milvus_ids, img_path) to mysql
        self.test_connection()
        add_data = ("INSERT INTO {} "
                    "(image_key, box, score) "
                    "VALUES (%s, %s, %s)".format(table_name))
        data = (image_key, box, score)
        try:
            self.cursor.execute(add_data, data)
            self.conn.commit()
            LOGGER.debug(f"MYSQL loads data to table: {table_name} successfully")
            return True
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {add_data}")
            return False

    def drop_table(self, table_name: str = DEFAULT_TABLE):
        # Delete mysql table if exists
        self.test_connection()
        sql = f"drop table if exists {table_name};"
        try:
            self.cursor.execute(sql)
            LOGGER.debug(f"MYSQL delete table:{table_name}")
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            raise Exception("MYSQL ERROR: {} with sql: {}".format(e, sql))

    def delete_all_data(self, table_name: str = DEFAULT_TABLE):
        # Delete all the data in mysql table
        self.test_connection()
        sql = f"delete from {table_name};"
        try:
            self.cursor.execute(sql)
            self.conn.commit()
            LOGGER.debug(f"MYSQL delete all data in table:{table_name}")
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            raise Exception("MYSQL ERROR: {} with sql: {}".format(e, sql))

    def count_table(self, table_name: str = DEFAULT_TABLE):
        # Get the number of mysql table
        self.test_connection()
        sql = "select count(milvus_id) from " + table_name + ";"
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            LOGGER.debug(f"MYSQL count table:{table_name}")
            return results[0][0]
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            raise Exception("MYSQL ERROR: {} with sql: {}".format(e, sql))


MYSQL_CLIENT = MysqlClient()
