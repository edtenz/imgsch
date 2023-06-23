from config import DEFAULT_TABLE
from milvus_helpers import MILVUS_CLIENT, insert_milvus_ops, search_milvus_ops
from model import Resnet50, VitTiny224, VitBase224


def test_resnet50_extract_features():
    model = Resnet50()
    obj_features = model.extract_features('../data/objects.png')
    for obj_feat in obj_features:
        print(obj_feat)
        assert len(obj_feat.features) == 2048


def test_vit224_extract_features():
    model = VitTiny224()
    obj_features = model.extract_features('../data/objects.png')
    for obj_feat in obj_features:
        print(obj_feat)
        assert len(obj_feat.features) == 192


def test_vitTiny224_extract_primary_features():
    model = VitTiny224()
    obj_feat = model.extract_primary_features('../data/objects.png')
    print(obj_feat)
    assert len(obj_feat.features) == 192


def test_vitBase224_extract_primary_features():
    model = VitBase224()
    obj_feat = model.extract_primary_features('../data/objects.png')
    print(obj_feat)
    assert len(obj_feat.features) == 768


def test_vit224_extract_pipeline():
    table_name = 'test_collection'
    dim = 768
    collection = MILVUS_CLIENT.create_collection(table_name, dim)
    print(f'A new collection created: {DEFAULT_TABLE}')

    model = VitBase224()
    p_insert = (
        model.pipeline()
        .map('vec', 'id', insert_milvus_ops(MILVUS_CLIENT, table_name))
        .output('url', 'key', 'sbox', 'label', 'score', 'id')
    )

    res = p_insert('../data/objects.png')
    size = res.size
    print(f'Insert {size} vectors')
    for i in range(size):
        it = res.get()
        print(f'{i}, url: {it[0]}, key: {it[1]}, sbox: {it[2]}, label: {it[3]}, score: {it[4]}, id: {it[5]}')

    print('Number of data inserted:', collection.num_entities)

    # Search pipeline
    p_search_pre = (
        model.pipeline()
        .flat_map('vec', ('pred', 'distance'), search_milvus_ops(MILVUS_CLIENT, table_name, 5))
        .output('sbox', 'pred', 'distance')
    )

    res = p_search_pre('../data/objects.png')
    print(f'Number of search results: {res.size}')
    for i in range(res.size):
        it = res.get()
        print(f'{i}, box:{it[0]}, pred: {it[1]}, distance: {it[2]}')
