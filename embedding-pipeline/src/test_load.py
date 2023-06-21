from load import do_load
from milvus_helpers import MILVUS_CLIENT
from minio_helpers import MINIO_CLIENT
from model import VitBase224
from mysql_helpers import MYSQL_CLIENT

MODEL = VitBase224()


def test_process():
    img_dir = '../data'
    count = do_load(img_dir, MODEL, MILVUS_CLIENT, MYSQL_CLIENT, MINIO_CLIENT)
    print(count)
