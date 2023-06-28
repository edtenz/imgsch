import os

from minio import Minio

from config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET_NAME, MINIO_DOWNLOAD_PATH
from image_helper import gen_file_key
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

    def list_objects(self, bucket_name: str = MINIO_BUCKET_NAME) -> list[str]:
        """
        List all objects in Minio bucket
        :param bucket_name: bucket name
        :return: list of objects in Minio bucket
        """
        objects = self.minio_client.list_objects(bucket_name, recursive=False)
        return [obj.object_name for obj in objects]

    def upload(self, object_name: str, file_path: str, bucket_name: str = MINIO_BUCKET_NAME) -> bool:
        """
        Upload object to Minio bucket
        :param object_name:  object name in Minio bucket
        :param file_path: path to file in local directory
        :param bucket_name: bucket name
        :return: True if upload successful, False otherwise
        """
        try:
            with open(file_path, 'rb') as f:
                LOGGER.debug(f'upload object {object_name} to: {object_name} from {file_path}')
                self.minio_client.put_object(
                    bucket_name,
                    object_name,
                    f,
                    os.stat(file_path).st_size
                )
                LOGGER.info(f'Image uploaded to {bucket_name}/{object_name}')
            return True
        except Exception as e:
            LOGGER.error(f'Failed to upload object {object_name} from {file_path}: {e}')
            return False

    def upload_file(self, file_path: str, bucket_name: str = MINIO_BUCKET_NAME) -> str:
        """
        Upload file to Minio bucket and return object name
        :param file_path: local file path
        :param bucket_name: bucket name
        :return: object name if upload successful, empty string otherwise
        """
        object_name = gen_file_key(file_path)
        if object_name == '':
            return ''
        if self.upload(object_name, file_path, bucket_name):
            return object_name
        else:
            return ''

    def exists_object(self, object_name: str, bucket_name: str = MINIO_BUCKET_NAME) -> bool:
        """
        Check if object exists in Minio bucket
        :param object_name: object name in Minio bucket
        :param bucket_name: bucket name
        :return: True if object exists, False otherwise
        """
        try:
            self.minio_client.stat_object(bucket_name, object_name)
            return True
        except Exception as e:
            LOGGER.error(f'Failed to check object {object_name}: {e}')
            return False

    def download(self, object_name: str, file_path: str, bucket_name: str = MINIO_BUCKET_NAME) -> bool:
        """
        Download object from Minio bucket to local directory
        :param object_name: object name in Minio bucket
        :param file_path: local file path
        :param bucket_name: bucket name
        :return: True if download successful, False otherwise
        """
        try:
            LOGGER.debug(f'download object {object_name} to {file_path}')
            self.minio_client.fget_object(bucket_name, object_name, file_path)
            LOGGER.info(f'Image downloaded to {file_path}')
            return True
        except Exception as e:
            LOGGER.error(f'Failed to download object {object_name} to {file_path}: {e}')
            return False


def upload_minio_ops(minio_cli: MinioClient, bucket_name: str = MINIO_BUCKET_NAME) -> callable:
    def wrapper(key, url) -> bool:
        if minio_cli.exists_object(key, bucket_name):
            return True
        return minio_cli.upload(key, url, bucket_name)

    return wrapper


def download_minio_ops(minio_cli: MinioClient,
                       bucket_name: str = MINIO_BUCKET_NAME,
                       download_dir: str = MINIO_DOWNLOAD_PATH) -> callable:
    def wrapper(object_name: str) -> str:
        if not minio_cli.exists_object(object_name, bucket_name):
            LOGGER.error(f'Object {object_name} does not exist in bucket {bucket_name}')
            return ''
        file_path = os.path.join(download_dir, object_name)
        if minio_cli.download(object_name, file_path, bucket_name):
            return file_path
        LOGGER.error(f'Failed to download object {object_name} from bucket {bucket_name}')
        return ''

    return wrapper


def download_object(minio_cli: MinioClient,
                    object_name: str,
                    download_dir: str = MINIO_DOWNLOAD_PATH) -> str:
    """
    Download object from Minio bucket to local file system
    Args:
        object_name: object name in Minio bucket
        minio_cli: minio client
        download_dir: download directory in local file system

    Returns: download path in local file system

    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    download_path = os.path.join(download_dir, object_name)
    ok = minio_cli.download(object_name, download_path)
    if ok:
        return download_path
    else:
        return ''


MINIO_CLIENT = MinioClient()
