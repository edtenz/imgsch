import os

############### Number of log files ###############
LOGS_NUM = int(os.getenv("logs_num", "0"))

############### Detectserver Configuration ###############
HTTP_PORT = os.getenv("HTTP_PORT", "8099")

############### Minio Configuration ###############
MINIO_ADDR = os.getenv("MINIO_HOST", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "imgsch")
MINIO_DOWNLOAD_PATH = os.getenv("MINIO_DOWNLOAD_PATH", "tmp/download-images")
