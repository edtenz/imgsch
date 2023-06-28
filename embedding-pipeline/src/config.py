import os

############### Number of log files ###############
LOGS_NUM = int(os.getenv("logs_num", "0"))

############### Detectserver Configuration ###############
HTTP_PORT = os.getenv("HTTP_PORT", "8090")

############### Minio Configuration ###############
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_DOWNLOAD_PATH = os.getenv("MINIO_DOWNLOAD_PATH", "tmp/download-images")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "imgsch")
MINIO_PROXY_ENDPOINT = os.getenv("MINIO_PROXY_ENDPOINT", "localhost:10086")

############### Milvus Configuration ###############
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))
INDEX_FILE_SIZE = int(os.getenv("INDEX_FILE_SIZE", "1024"))
INDEX_TYPE = os.getenv("INDEX_TYPE", "IVF_SQ8")
METRIC_TYPE = os.getenv("METRIC_TYPE", "IP")
NLIST = int(os.getenv("NLIST", "8192"))
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "milvus_imgsch_tab")
TOP_K = int(os.getenv("TOP_K", "10"))

############### MySQL Configuration ###############
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PWD = os.getenv("MYSQL_PWD", "helloworld123")
MYSQL_DB = os.getenv("MYSQL_DB", "imgsch_db")

############### Data Path ###############
UPLOAD_PATH = os.getenv("UPLOAD_PATH", "tmp/search-images")
