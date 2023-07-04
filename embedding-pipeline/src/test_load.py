from towhee import pipe

from load import do_embedding
from milvus_helpers import MILVUS_CLIENT
from minio_helpers import MINIO_CLIENT
from model import VitBase224
from model import extract_features_ops
from mysql_helpers import MYSQL_CLIENT

MODEL = VitBase224()


def test_process():
    img_dir = '../data'
    table_name = 'test_collection'
    test_bucket_name = 'mybucket'
    dim = 768
    vit_model = VitBase224()
    count = do_embedding(img_dir, vit_model, MILVUS_CLIENT, MYSQL_CLIENT, MINIO_CLIENT, test_bucket_name, table_name,
                         dim)
    print(count)


def test_load():
    insert_p = (
        pipe.input('url')
        .map('url', 'key', lambda url: url.split('/')[-1])
        .map('url', ('sbox', 'label', 'score', 'vec'), extract_features_ops(MODEL))
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
