from pymilvus import FieldSchema, CollectionSchema, DataType, Collection, utility, connections
from towhee import ops

from config import MILVUS_HOST, MILVUS_PORT, DEFAULT_TABLE
from model import Resnet50, Vit224


def test_resnet50_extract_features():
    model = Resnet50()
    obj_features = model.extract_features('../data/objects.png')
    for obj_feat in obj_features:
        print(obj_feat)
        assert len(obj_feat.features) == 2048


def test_vit224_extract_features():
    model = Vit224()
    obj_features = model.extract_features('../data/objects.png')
    for obj_feat in obj_features:
        print(obj_feat)
        assert len(obj_feat.features) == 192


def test_vit224_extract_primary_features():
    model = Vit224()
    obj_feat = model.extract_primary_features('../data/objects.png')
    print(obj_feat)
    assert len(obj_feat.features) == 192


def create_milvus_collection(collection_name: str = DEFAULT_TABLE, dim: int = 192):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name='key', dtype=DataType.VARCHAR, description='image key', max_length=40,
                    is_primary=True, auto_id=False),
        FieldSchema(name='sbox', dtype=DataType.VARCHAR, description='image box', max_length=20),
        FieldSchema(name='vec', dtype=DataType.FLOAT_VECTOR, description='image embedding vectors', dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description='reverse image search')
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        'metric_type': 'L2',
        'index_type': 'IVF_SQ8',
        'params': {"nlist": dim}
    }
    collection.create_index(field_name='vec', index_params=index_params)
    return collection


def test_vit224_extract_pipeline():
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

    collection = create_milvus_collection(DEFAULT_TABLE, 192)
    print(f'A new collection created: {DEFAULT_TABLE}')

    model = Vit224()
    p_insert = (
        model.pipeline()
        .map(('key', 'sbox', 'vec'), 'mr', ops.ann_insert.milvus_client(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            collection_name=DEFAULT_TABLE,
        ))
        .output('key', 'sbox', 'mr')
    )
    res = p_insert('../data/objects.png')
    size = res.size
    print(f'Insert {size} vectors')
    for i in range(size):
        it = res.get()
        print(f'{i}, key: {it[0]}, box: {it[1]}, mr: {it[2]}')

    print('Number of data inserted:', collection.num_entities)

    # Search pipeline
    p_search_pre = (
        model.pipeline()
        .map('vec', ('search_res'), ops.ann_search.milvus_client(
            host=MILVUS_HOST, port=MILVUS_PORT, limit=10,
            collection_name=DEFAULT_TABLE))
        .map('search_res', ('pred', 'distance'), lambda x: ([y[0] for y in x], [y[1] for y in x]))
        .output('pred', 'distance')
    )
    collection.load()
    res = p_search_pre('../data/fruit.png')
    print(f'Number of search results: {res.size}')
    for i in range(res.size):
        it = res.get()
        print(f'{i}, pred: {it[0]}, distance: {it[1]}')
