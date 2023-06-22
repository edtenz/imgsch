from config import DEFAULT_TABLE
from milvus_helpers import MILVUS_CLIENT
from model import VitTiny224


def test_insert():
    table_name = "test_collection"
    dim = 192
    top_k = 10
    MILVUS_CLIENT.delete_collection(table_name)
    MILVUS_CLIENT.create_collection(table_name, dim)

    vit_model = VitTiny224()
    obj_feat = vit_model.extract_primary_features('../data/objects.png')
    if obj_feat is None:
        print('extract feature failed')
        return
    print('vector size:', len(obj_feat.features))

    id = MILVUS_CLIENT.insert(table_name, obj_feat.features)
    print('insert vec id:', id)
    res = MILVUS_CLIENT.search_vectors(table_name, obj_feat.features, top_k)
    # print('search res:', res)
    for item in res:
        print(item)


def test_search_vectors():
    vit_model = VitTiny224()
    obj_feat = vit_model.extract_primary_features('../data/test.jpg')
    print('vector size:', len(obj_feat.features))
    res = MILVUS_CLIENT.search_vectors(DEFAULT_TABLE, obj_feat.features, 10)
    print(res)


def test_delete_collection():
    res = MILVUS_CLIENT.delete_collection(DEFAULT_TABLE)
    print(res)
