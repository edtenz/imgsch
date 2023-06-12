import hashlib
import os
import sys

from config import DEFAULT_TABLE
from logs import LOGGER
from milvus_helpers import MilvusHelper
from minio_helpers import MinioHelper
from model import Resnet50
from mysql_helpers import MySQLHelper


# Get the path to the image
def get_imgs(path):
    pics = []
    for f in os.listdir(path):
        if ((f.endswith(extension) for extension in
             ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']) and not f.startswith('.DS_Store')):
            pics.append(os.path.join(path, f))
    return pics


# Get the vector of images
def extract_features(img_dir: str, model: Resnet50) -> (list[float], list[str]):
    try:
        feats = []
        names = []
        img_list = get_imgs(img_dir)
        total = len(img_list)
        for i, img_path in enumerate(img_list):
            try:
                norm_feat = model.extract_features(img_path)
                feats.append(norm_feat)
                names.append(calculate_md5(img_path))
                print(f"Extracting feature from image No. {i + 1} , {total} images in total")
            except Exception as e:
                LOGGER.error(f"Error with extracting feature from image {e}")
                continue
        return feats, names
    except Exception as e:
        LOGGER.error(f"Error with extracting feature from image {e}")
        sys.exit(1)


# Combine the id of the vector and the name of the image into a list
def format_data(ids, names):
    data = []
    for i in range(len(ids)):
        value = (str(ids[i]), names[i])
        data.append(value)
    return data


def calculate_md5(file_path: str) -> str:
    """
    Calculate MD5 hash of file
    :param file_path: path to file
    :return: md5 hash of file
    """
    try:
        with open(file_path, 'rb') as f:
            file_content = f.read()
            if file_content:
                md5_hash = hashlib.md5(file_content).hexdigest()
                return md5_hash
            else:
                print(f"File '{file_path}' is empty.")
                return ''
    except Exception as e:
        print(f"Error calculating MD5 hash of file '{file_path}': {str(e)}")
        return ''


def do_load(table_name, image_dir, model, milvus_client, mysql_cli, minio_cli) -> int:
    if not table_name:
        table_name = DEFAULT_TABLE

    milvus_client.create_index(table_name)
    mysql_cli.create_mysql_table(table_name)

    img_list = get_imgs(image_dir)
    total = len(img_list)
    success_count = 0
    for i, img_path in enumerate(img_list):
        LOGGER.info(f"Process file {img_path}, {i + 1}/{total}")
        count = process(img_path, model, milvus_client, mysql_cli, minio_cli, table_name)
        success_count += count
        LOGGER.info(f"Process file {img_path} successfully, succ count: {success_count}/{total}")

    return success_count


def process(img_path: str,
            model: Resnet50,
            milvus_client: MilvusHelper,
            mysql_cli: MySQLHelper,
            minio_cli: MinioHelper,
            table_name=DEFAULT_TABLE) -> int:
    object_name = minio_cli.upload_file(img_path)
    if object_name == '':
        LOGGER.warn(f"Upload file failed: {img_path}")
        return 0
    feat = model.extract_features(img_path)
    if len(feat) == 0:
        LOGGER.warn(f"Extract feature failed: {img_path}")
        return 0
    vectors = [feat]
    ids = milvus_client.insert(table_name, vectors)
    if len(ids) == 0:
        LOGGER.warn(f"Insert vectors failed: {img_path}")
        return 0
    names = [object_name]
    mysql_cli.load_data_to_mysql(table_name, format_data(ids, names))
    LOGGER.debug(f"Process file {img_path} successfully")
    return len(ids)
