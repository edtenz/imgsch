from towhee import pipe

from load import do_milvus_embedding
from milvus_helpers import MilvusClient
from model import VitBase224
from model import extract_features_ops
from mysql_helpers import MysqlClient


def test_process():
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
