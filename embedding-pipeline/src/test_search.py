from config import DEFAULT_TABLE
from milvus_helpers import MilvusClient
from minio_helpers import MinioClient
from model import VitBase224
from mysql_helpers import MysqlClient
from search import do_search, do_download

MYSQL_CLIENT = MysqlClient()
MINIO_CLIENT = MinioClient()
MILVUS_CLIENT = MilvusClient()


def test_do_search():
    img_path = '../data/objects.png'
    table_name = DEFAULT_TABLE
    vit_model = VitBase224()
    obj_feat, candidate_box, res_list = do_search(img_path, vit_model, MILVUS_CLIENT, MYSQL_CLIENT, table_name)
    print(obj_feat)
    print(candidate_box)
    print(res_list)


def test_do_download():
    image_key = '44cffb7fe6339ad06e4f046ae52fa987.jpg'
    img_path = do_download(image_key, MINIO_CLIENT)
    print(img_path)


def test_do_search2():
    img_url = 'http://localhost:10086/file/imgsch/008e7a0c7582d987b183a59133f7169e.jpg'
    obj, candidate_box, res = do_search(img_url, VitBase224(), MILVUS_CLIENT, MYSQL_CLIENT, DEFAULT_TABLE)
    print(obj)
    print(candidate_box)
    print('results size:', len(res))
    for it in res:
        print("res:", it)
