from config import (
    DEFAULT_TABLE,
    ES_HOST,
    ES_PORT,
    ES_INDEX,
)
from milvus_helpers import MilvusClient
from model import VitBase224
from mysql_helpers import MysqlClient
from search import EsClient, do_es_search
from search import do_milvus_search


def test_do_milvus_search_local():
    img_path = '../data/objects.png'
    milvus_cli = MilvusClient()
    mysql_cli = MysqlClient()
    table_name = DEFAULT_TABLE
    vit_model = VitBase224()
    obj_feat, candidate_box, res_list = do_milvus_search(img_path, vit_model, milvus_cli, mysql_cli, table_name)
    print(obj_feat)
    print(candidate_box)
    print(res_list)


def test_do_milvus_search_url():
    img_url = 'http://localhost:10086/file/imgsch/008e7a0c7582d987b183a59133f7169e.jpg'
    milvus_cli = MilvusClient()
    mysql_cli = MysqlClient()
    obj, candidate_box, res = do_milvus_search(img_url, VitBase224(), milvus_cli, mysql_cli, DEFAULT_TABLE)
    print(obj)
    print(candidate_box)
    print('results size:', len(res))
    for it in res:
        print("res:", it)


def test_do_es_search():
    es_cli = EsClient(host=ES_HOST, port=ES_PORT)
    index_name = ES_INDEX
    img_url = 'http://localhost:10086/file/imgsch/07dd1974a83862447b3dfa23957a4cfc.jpg'
    vit_model = VitBase224()

    obj_feat, candidates, res = do_es_search(img_url, vit_model, es_cli, index_name)
    print(obj_feat)
    print(candidates)
    print(res)
