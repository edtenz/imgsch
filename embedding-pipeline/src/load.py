import sys

from pymilvus import Collection
from towhee import ops

import image_helper
from config import DEFAULT_TABLE, MILVUS_HOST, MILVUS_PORT, VECTOR_DIMENSION, MINIO_BUCKET_NAME
from logger import LOGGER
from milvus_helpers import MilvusClient
from minio_helpers import MinioClient
from model import Model
from mysql_helpers import MysqlClient


def extract_features(img_dir: str, model: Model) -> (list[float], list[str]):
    """
    Extract features from images
    :param img_dir:
    :param model:
    :return:
    """
    try:
        feats = []
        names = []
        img_list = image_helper.get_images(img_dir)
        total = len(img_list)
        for i, img_path in enumerate(img_list):
            try:
                norm_feat = model.extract_features(img_path)
                feats.append(norm_feat)
                names.append(image_helper.md5_file(img_path))
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


def do_load(
        image_dir: str,
        model: Model,
        milvus_client: MilvusClient,
        mysql_cli: MysqlClient,
        minio_cli: MinioClient,
        table_name: str = DEFAULT_TABLE,
        dim: int = VECTOR_DIMENSION) -> int:
    if not table_name:
        table_name = DEFAULT_TABLE

    collection = milvus_client.create_collection(table_name, dim)
    LOGGER.info(f"Collection information: {table_name}")

    mysql_cli.create_table(table_name)
    LOGGER.info(f"Table information: {table_name}")

    minio_cli.create_bucket(MINIO_BUCKET_NAME)
    LOGGER.info(f"Bucket information: {MINIO_BUCKET_NAME}")

    img_list = image_helper.get_images(image_dir)
    total = len(img_list)
    LOGGER.info(f"Start to process {total} files")
    success_count = 0
    for i, img_path in enumerate(img_list):
        LOGGER.info(f"Process file {img_path}, {i + 1}/{total}")
        ok = process(img_path, model, milvus_client, mysql_cli, minio_cli, collection, table_name)
        if ok:
            success_count += 1

        LOGGER.info(f"Process file {img_path} successfully, succ count: {success_count}/{total}")

    return success_count


def process(img_path: str,
            model: Model,
            milvus_client: MilvusClient,
            mysql_cli: MysqlClient,
            minio_cli: MinioClient,
            collection: Collection,
            table_name: str = DEFAULT_TABLE) -> bool:
    p_insert = (
        model.pipeline()
        .map(('key', 'url'), 'upload_res', minio_cli.upload)
        .map(('key', 'vec'), 'mr', ops.ann_insert.milvus_client(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            collection_name=table_name,
        ))
        .map(('key', 'sbox', 'score'), 'db_res', mysql_cli.insert_into)
        .output('url', 'key', 'sbox', 'label', 'score', 'mr', 'upload_res', 'db_res')
    )
    res = p_insert(img_path)
    size = res.size
    print(f'Insert {size} vectors')
    for i in range(size):
        it = res.get()
        print(
            f'{i}, url: {it[0]}, key: {it[1]}, sbox: {it[2]}, label: {it[3]}, score: {it[4]}, '
            f'mr: {it[5]}, upload_res: {it[6]}, db_res: {it[7]}')

    print('Number of data inserted:', collection.num_entities)
    collection.load()
    LOGGER.debug(f"Process file {img_path} successfully")
    return True
