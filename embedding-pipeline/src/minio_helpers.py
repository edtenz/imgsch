import os

from minio import Minio

from config import MINIO_BUCKET_NAME, MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY
from image_helper import md5_file
from logger import LOGGER


class MinioClient(object):
    def __init__(self, endpoint=MINIO_ENDPOINT,
                 access_key=MINIO_ACCESS_KEY,
                 secret_key=MINIO_SECRET_KEY,
                 secure=False,
                 bucket_name=MINIO_BUCKET_NAME):
        self.bucket_name = bucket_name
        self.minio_client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )

        if not self.minio_client.bucket_exists(self.bucket_name):
            self.minio_client.make_bucket(bucket_name)
            LOGGER.info(f'Bucket {bucket_name} created successfully')
        else:
            LOGGER.info(f'Bucket {bucket_name} already exists')

    def upload(self, object_name: str, file_path: str) -> bool:
        """
        Upload object to Minio bucket
        :param object_name:  object name in Minio bucket
        :param file_path: path to file in local directory
        :return: True if upload successful, False otherwise
        """
        try:
            with open(file_path, 'rb') as f:
                LOGGER.debug(f'upload object to: {object_name} from {file_path}')
                self.minio_client.put_object(
                    self.bucket_name,
                    object_name,
                    f,
                    os.stat(file_path).st_size
                )
                LOGGER.info(f'Image uploaded to {self.bucket_name}/{object_name}')
            return True
        except Exception as e:
            LOGGER.error(f'Failed to upload object {object_name} from {file_path}: {e}')
            return False

    def upload_file(self, file_path: str) -> str:
        """
        Upload file to Minio bucket and return object name
        :param file_path: local file path
        :return: object name if upload successful, empty string otherwise
        """
        object_name = md5_file(file_path)
        if object_name == '':
            return ''
        if self.upload(object_name, file_path):
            return object_name
        else:
            return ''


MINIO_CLIENT = MinioClient()
