import json

import requests

from config import DEFAULT_TABLE, VECTOR_DIMENSION, MINIO_PROXY_ENDPOINT
from logger import LOGGER
from milvus_helpers import MilvusClient, insert_milvus_ops
from model import ImageFeatureModel
from mysql_helpers import MysqlClient, insert_mysql_ops


def do_embedding(
        bucket_name: str,
        model: ImageFeatureModel,
        milvus_client: MilvusClient,
        mysql_cli: MysqlClient,
        table_name: str = DEFAULT_TABLE,
        dim: int = VECTOR_DIMENSION) -> int:
    collection = milvus_client.create_collection(table_name, dim)
    LOGGER.info(f"Collection information: {table_name}")

    mysql_cli.create_table(table_name)
    LOGGER.info(f"Table information: {table_name}")

    lst_url = f'http://{MINIO_PROXY_ENDPOINT}/file/{bucket_name}'
    LOGGER.info(f"List url: {lst_url}")
    response = requests.get(lst_url)
    LOGGER.debug(f"Response: {response}")
    response.raise_for_status()  # Raise an exception if the request was unsuccessful

    object_names = json.loads(response.content)
    total = len(object_names)
    LOGGER.info(f"Start to process {total} files")
    success_count = 0
    for i, object_name in enumerate(object_names):
        LOGGER.info(f"Process file {object_name}, {i + 1}/{total}")
        img_url = f'http://{MINIO_PROXY_ENDPOINT}/api/{bucket_name}/{object_name}'
        try:
            ok = embedding_pipeline(img_url, model, milvus_client, mysql_cli, table_name)
            if ok:
                success_count += 1

            LOGGER.info(f"Process file {object_name} successfully, succ count: {success_count}/{total}")
        except Exception as e:
            LOGGER.error(f"Process file {object_name} failed: {e}")
            continue

    LOGGER.info(f"Process {success_count} files successfully, total: {total}")
    LOGGER.info(f"Load {collection.num_entities} entities rows")

    return success_count


def embedding_pipeline(img_url: str,
                       model: ImageFeatureModel,
                       milvus_client: MilvusClient,
                       mysql_cli: MysqlClient,
                       table_name: str = DEFAULT_TABLE) -> bool:
    p_insert = (
        model.pipeline()
        .map('vec', 'id', insert_milvus_ops(milvus_client, table_name))
        .map(('id', 'url', 'sbox', 'score', 'label'), 'db_res', insert_mysql_ops(mysql_cli, table_name))
        .output('url', 'sbox', 'label', 'score', 'id', 'db_res')
    )
    LOGGER.debug(f"Process file from url: {img_url}")
    res = p_insert(img_url)
    size = res.size
    print(f'Insert {size} vectors')
    for i in range(size):
        it = res.get()
        print(
            f'{i}, url: {it[0]}, sbox: {it[1]}, label: {it[2]}, score: {it[3]}, '
            f'id: {it[4]}, db_res: {it[5]}')

    LOGGER.debug(f"Process file {img_url} successfully")
    return True
