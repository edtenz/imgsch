from config import DEFAULT_TABLE
from milvus_helpers import MILVUS_CLIENT
from minio_helpers import MINIO_CLIENT
from model import VitBase224
from mysql_helpers import MYSQL_CLIENT
from search import do_search, do_download


def test_do_search():
    img_path = '../data/objects.png'
    table_name = DEFAULT_TABLE
    vit_model = VitBase224()
    obj_feat, res_list = do_search(img_path, vit_model, MILVUS_CLIENT, MYSQL_CLIENT, table_name)
    print(obj_feat)
    print(res_list)


def test_do_download():
    image_key = '44cffb7fe6339ad06e4f046ae52fa987.jpg'
    img_path = do_download(image_key, MINIO_CLIENT)
    print(img_path)
