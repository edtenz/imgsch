from towhee import ops

from config import MILVUS_HOST, MILVUS_PORT, DEFAULT_TABLE
from milvus_helpers import MILVUS_CLIENT
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
    collection = MILVUS_CLIENT.create_collection(DEFAULT_TABLE, 768)
    print(f'A new collection created: {DEFAULT_TABLE}')

    model = VitBase224()
    p_insert = (
        model.pipeline()
        .map(('key', 'vec'), 'mr', ops.ann_insert.milvus_client(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            collection_name=DEFAULT_TABLE,
        ))
        .output('url', 'key', 'sbox', 'label', 'score', 'mr')
    )
    res = p_insert('../data/objects.png')
    size = res.size
    print(f'Insert {size} vectors')
    for i in range(size):
        it = res.get()
        print(f'{i}, url: {it[0]}, key: {it[1]}, sbox: {it[2]}, label: {it[3]}, score: {it[4]}, mr: {it[5]}')

    print('Number of data inserted:', collection.num_entities)
    collection.load()

    # Search pipeline
    p_search_pre = (
        model.pipeline()
        .map('vec', ('search_res'), ops.ann_search.milvus_client(
            host=MILVUS_HOST, port=MILVUS_PORT, limit=10,
            collection_name=DEFAULT_TABLE))
        .map('search_res', ('pred', 'distance'), lambda x: ([y[0] for y in x], [y[1] for y in x]))
        .output('pred', 'distance', 'search_res')
    )

    res = p_search_pre('../data/objects.png')
    print(f'Number of search results: {res.size}')
    for i in range(res.size):
        it = res.get()
        print(f'{i}, search_res: {it[2]}')
