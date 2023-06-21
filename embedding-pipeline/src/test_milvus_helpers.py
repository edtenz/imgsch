from milvus_helpers import MILVUS_CLIENT
from config import DEFAULT_TABLE
from model import VitTiny224


def test_insert():
    if not MILVUS_CLIENT.has_collection(DEFAULT_TABLE):
        MILVUS_CLIENT.create_collection(DEFAULT_TABLE, 192)

    vit_model = VitTiny224()
    obj_feat = vit_model.extract_primary_features('../data/test.jpg')
    print('vector size:', len(obj_feat.features))

    res = MILVUS_CLIENT.insert(DEFAULT_TABLE, obj_feat.features)
    print('insert res:', res)
    res = MILVUS_CLIENT.search_vectors(DEFAULT_TABLE, obj_feat.features, 10)
    print('search res:', res)


def test_search_vectors():
    vit_model = VitTiny224()
    obj_feat = vit_model.extract_primary_features('../data/test.jpg')
    print('vector size:', len(obj_feat.features))
    res = MILVUS_CLIENT.search_vectors(DEFAULT_TABLE, obj_feat.features, 10)
    print(res)


def test_delete_collection():
    res = MILVUS_CLIENT.delete_collection(DEFAULT_TABLE)
    print(res)
