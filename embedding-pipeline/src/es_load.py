import json

import requests
from towhee import pipe

from config import ES_INDEX, MINIO_PROXY_ENDPOINT
from es_helpers import EsClient, insert_img_doc_ops, create_img_index
from logger import LOGGER
from model import ImageFeatureModel, extract_features_ops


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
