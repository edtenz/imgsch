import os

from config import MINIO_DOWNLOAD_PATH
from detect import Detector, BoundingBox
from extract import Extractor
from minio_helpers import MinioClient, download_object


def do_detect(key: str,
              minio_cli: MinioClient,
              detector: Detector,
              download_dir: str = MINIO_DOWNLOAD_PATH) -> list[BoundingBox]:
    """
    Detect objects in image
    Args:
        key: object name in Minio bucket
        minio_cli: minio client
        detector: detector
        download_dir: download directory in local file system

    Returns: bounding boxes

    """

    download_path = download_object(key, minio_cli, download_dir)
    if not download_path:
        return []

    bboxes = detector.detect(download_path)
    os.remove(download_path)
    return bboxes


def do_extract(key: str,
               minio_cli: MinioClient,
               extractor: Extractor,
               box: tuple[int, int, int, int],
               download_dir: str = MINIO_DOWNLOAD_PATH) -> list[float]:
    """
    Extract features from image
    Args:
        key: object name in Minio bucket
        minio_cli: minio client
        extractor: extractor
        box: box to extract features
        download_dir: download directory in local file system

    Returns: image features in box

    """

    download_path = download_object(key, minio_cli, download_dir)
    if not download_path:
        return []

    features = extractor.extract(download_path, box)
    os.remove(download_path)
    return features
