import json

import requests
from towhee import pipe

from config import DEFAULT_TABLE, VECTOR_DIMENSION
from config import ES_INDEX, MINIO_PROXY_ENDPOINT
from es_helpers import EsClient, insert_img_doc_ops, create_img_index
from logger import LOGGER
from milvus_helpers import MilvusClient, insert_milvus_ops
from model import ImageFeatureModel, extract_features_ops
from mysql_helpers import MysqlClient, insert_mysql_ops


def do_milvus_embedding(
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
        img_url = f'http://{MINIO_PROXY_ENDPOINT}/file/{bucket_name}/{object_name}'
        try:
            ok = embedding_milvus_pipe(img_url, model, milvus_client, mysql_cli, table_name)
            if ok:
                success_count += 1

            LOGGER.info(f"Process file {object_name} successfully, succ count: {success_count}/{total}")
        except Exception as e:
            LOGGER.error(f"Process file {object_name} failed: {e}")
            continue

    LOGGER.info(f"Process {success_count} files successfully, total: {total}")
    LOGGER.info(f"Load {collection.num_entities} entities rows")

    return success_count


def embedding_milvus_pipe(img_url: str,
                          model: ImageFeatureModel,
                          milvus_client: MilvusClient,
                          mysql_cli: MysqlClient,
                          table_name: str = DEFAULT_TABLE) -> bool:
    p_insert = (
        model.pipeline()
        .filter(('vec',), ('vec',), 'vec', lambda x: x is not None and len(x) > 0)
        .map('vec', 'id', insert_milvus_ops(milvus_client, table_name))
        .filter(('id',), ('id',), 'id', lambda x: x is not None and len(x) > 0)
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


def do_es_embedding(
        bucket_name: str,
        model: ImageFeatureModel,
        es_cli: EsClient,
        index_name: str = ES_INDEX,
        max_count: int = 0) -> int:
    create_img_index(es_cli, index_name)

    lst_url = f'http://{MINIO_PROXY_ENDPOINT}/file/{bucket_name}'
    LOGGER.info(f"List url: {lst_url}")
    response = requests.get(lst_url)
    LOGGER.debug(f"Response: {response}")
    response.raise_for_status()  # Raise an exception if the request was unsuccessful

    object_names = json.loads(response.content)
    if 0 < max_count < len(object_names):
        object_names = object_names[:max_count]

    total = len(object_names)
    LOGGER.info(f"Start to process {total} files")
    success_count = 0
    for i, object_name in enumerate(object_names):
        LOGGER.info(f"Process file {object_name}, {i + 1}/{total}")
        img_url = f'http://{MINIO_PROXY_ENDPOINT}/file/{bucket_name}/{object_name}'
        try:
            ok = embedding_es_pipe(img_url, model, es_cli, index_name)
            if ok:
                success_count += 1

            LOGGER.info(f"Process file {object_name} successfully, succ count: {success_count}/{total}")
        except Exception as e:
            LOGGER.error(f"Process file {object_name} failed: {e}")
            continue

    return success_count


def embedding_es_pipe(img_url: str,
                      model: ImageFeatureModel,
                      es_cli: EsClient,
                      index_name: str = ES_INDEX) -> bool:
    p_insert = (
        pipe.input('url')
        .map('url', 'key', lambda x: x.split('/')[-1].split('.')[0])
        .filter(('key',), ('key',), 'key', lambda x: x is not None and len(x) > 0)
        .map('url', ('sbox', 'label', 'score', 'features'), extract_features_ops(model))
        .filter(('features',), ('features',), 'features', lambda x: x is not None and len(x) > 0)
        .map('features', 'fsize', lambda x: len(x))
        .map(('key', 'url', 'sbox', 'score', 'label', 'features', 'key'), 'res', insert_img_doc_ops(es_cli, index_name))
        .output('key', 'url', 'sbox', 'score', 'label', 'fsize', 'key', 'res')
    )

    res = p_insert(img_url)
    size = res.size
    if size == 0:
        LOGGER.info("no result")
        return False
    for i in range(size):
        it = res.get()
        LOGGER.info(f'inserted: {it}')
    return True
