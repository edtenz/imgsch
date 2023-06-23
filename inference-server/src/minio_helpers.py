import os

from minio import Minio

from config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET_NAME, MINIO_DOWNLOAD_PATH
from logger import LOGGER


class MinioClient(object):
    def __init__(self, endpoint=MINIO_ENDPOINT,
                 access_key=MINIO_ACCESS_KEY,
                 secret_key=MINIO_SECRET_KEY,
                 secure=False,
                 bucket_name=MINIO_BUCKET_NAME):
        self.bucket_name = bucket_name
        self.minio_client = Minio(endpoint=endpoint,
                                  access_key=access_key,
                                  secret_key=secret_key,
                                  secure=secure)

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

    def download(self, object_name: str, download_path: str) -> bool:
        """
        Download object from Minio bucket to local directory
        :param object_name: object name in Minio bucket
        :param download_path: download path in local directory
        :return: True if download successful, False otherwise
        """
        LOGGER.debug(f'Downloading object {object_name} from bucket {self.bucket_name} to {download_path}')
        try:
            self.minio_client.fget_object(self.bucket_name, object_name, download_path)
            LOGGER.info(f'Object downloaded to {download_path}')
            return True
        except Exception as e:
            LOGGER.error(
                f'Failed to download object {object_name} from bucket {self.bucket_name} to {download_path}: {e}')
            return False


def download_object(object_name: str,
                    minio_cli: MinioClient,
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


def remove_local_object(object_name: str, download_dir=MINIO_DOWNLOAD_PATH) -> bool:
    """
    Remove object from local file system
    :param object_name: object name in Minio bucket
    :param download_dir: download directory in local file system
    :return:  success or failure
    """
    try:
        fpath = os.path.join(download_dir, object_name)
        os.remove(fpath)
        return True
    except Exception as e:
        LOGGER.error(f'Failed to remove object {object_name} from {download_dir}: {e}')
        return False


MINIO_CLIENT = MinioClient()
