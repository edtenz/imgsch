import os

from minio import Minio

from config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET_NAME
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

        self.create_bucket(bucket_name)

    def create_bucket(self, bucket_name: str = MINIO_BUCKET_NAME) -> bool:
        """
        Create bucket in Minio
        :param bucket_name: bucket name
        :return: True if bucket created successfully, False otherwise
        """
        if self.minio_client.bucket_exists(bucket_name):
            LOGGER.info(f'Bucket {bucket_name} already exists')
            return True
        self.minio_client.make_bucket(bucket_name)
        LOGGER.info(f'Bucket {bucket_name} created successfully')
        return True

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
                LOGGER.info(f'Image uploaded to {self.bucket_name,}/{object_name}')
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

    def exists_object(self, object_name: str) -> bool:
        """
        Check if object exists in Minio bucket
        :param object_name: object name in Minio bucket
        :return: True if object exists, False otherwise
        """
        try:
            self.minio_client.stat_object(self.bucket_name, object_name)
            return True
        except Exception as e:
            LOGGER.error(f'Failed to check object {object_name}: {e}')
            return False

    def download(self, object_name: str, file_path: str) -> bool:
        """
        Download object from Minio bucket to local directory
        :param object_name: object name in Minio bucket
        :param file_path: local file path
        :return: True if download successful, False otherwise
        """
        try:
            LOGGER.debug(f'download object {object_name} to {file_path}')
            self.minio_client.fget_object(self.bucket_name, object_name, file_path)
            LOGGER.info(f'Image downloaded to {file_path}')
            return True
        except Exception as e:
            LOGGER.error(f'Failed to download object {object_name} to {file_path}: {e}')
            return False


MINIO_CLIENT = MinioClient()
