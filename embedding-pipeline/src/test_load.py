from load import do_load
from milvus_helpers import MILVUS_CLIENT
from minio_helpers import MINIO_CLIENT
from model import VitBase224
from mysql_helpers import MYSQL_CLIENT

MODEL = VitBase224()


def test_process():
    img_dir = '../data'
    table_name = 'test_collection'
    test_bucket_name = 'mybucket'
    dim = 768
    vit_model = VitBase224()
    count = do_load(img_dir, vit_model, MILVUS_CLIENT, MYSQL_CLIENT, MINIO_CLIENT, test_bucket_name, table_name, dim)
    print(count)
