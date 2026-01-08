import os
from dotenv import load_dotenv
from mysql.connector import pooling

load_dotenv()

DB_NAME = os.getenv("MYSQL_DB")
if not DB_NAME:
    raise RuntimeError("MYSQL_DB is not set in .env")

DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": DB_NAME,
}

POOL_SIZE = int(os.getenv("MYSQL_POOL_SIZE", 10))

pool = pooling.MySQLConnectionPool(
    pool_name="api_pool",
    pool_size=POOL_SIZE,
    pool_reset_session=True,
    **DB_CONFIG
)

def get_connection():
    return pool.get_connection()
