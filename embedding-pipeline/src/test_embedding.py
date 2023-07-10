from towhee import pipe

from config import ES_HOST, ES_PORT, ES_INDEX, MINIO_BUCKET_NAME
from embedding import (
    do_milvus_embedding,
    embedding_es_pipe,
    do_es_embedding,
)
from es_helpers import EsClient, create_img_index
from milvus_helpers import MilvusClient
from model import VitBase224
from model import extract_features_ops
from mysql_helpers import MysqlClient


def test_embedding_es_pipe():
    es_cli = EsClient(host=ES_HOST, port=ES_PORT)
    index_name = ES_INDEX
    ok = create_img_index(es_cli, index_name)
    print("ok: ", ok)
    if not ok:
        return

    img_url = 'http://localhost:10086/file/imgsch/0ad09b3fdb4fbd661923fb5555c7b341.jpg'
    model = VitBase224()
    res = embedding_es_pipe(img_url, model, es_cli, index_name)
    print("res: ", res)


def test_do_es_embedding():
    es_cli = EsClient(host=ES_HOST, port=ES_PORT)
    index_name = ES_INDEX
    model = VitBase224()
    test_count = 10000
    suc = do_es_embedding(MINIO_BUCKET_NAME, model, es_cli, index_name, test_count)
    print("suc: ", suc)


def test_do_milvus_embedding():
    vit_model = VitBase224()
    mysql_cli = MysqlClient()
    milvus_cli = MilvusClient()

    table_name = 'test_collection'
    test_bucket_name = 'mybucket'
    dim = 768
    count = do_milvus_embedding(test_bucket_name, vit_model, milvus_cli, mysql_cli, table_name, dim)
    print(count)


def test_load():
    vit_model = VitBase224()

    insert_p = (
        pipe.input('url')
        .map('url', 'key', lambda url: url.split('/')[-1])
        .map('url', ('sbox', 'label', 'score', 'vec'), extract_features_ops(vit_model))
        .filter(('vec',), ('vec',), 'vec', lambda vec: vec is not None and len(vec) > 0)
        .output('key', 'sbox', 'label', 'score', 'vec')
    )

    img_url = 'http://localhost:10086/file/search/e50997c27485836e7af3e9466ef662dc'
    res = insert_p(img_url)
    size = res.size
    if size == 0:
        print("no result")
        return
    for i in range(size):
        it = res.get()
        print(it)
