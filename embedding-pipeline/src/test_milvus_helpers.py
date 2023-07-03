from config import DEFAULT_TABLE
from milvus_helpers import MILVUS_CLIENT, search_milvus_ops
from model import VitTiny224, VitBase224


def test_insert():
    table_name = "test_collection"
    dim = 192
    top_k = 10
    MILVUS_CLIENT.delete_collection(table_name)
    MILVUS_CLIENT.create_collection(table_name, dim)

    vit_model = VitTiny224()
    obj_feat, candidate_box = vit_model.extract_primary_features('../data/objects.png')
    print('candidate_box:', candidate_box)
    if obj_feat is None:
        print('extract feature failed')
        return
    print('vector size:', len(obj_feat.features))

    vec_id = MILVUS_CLIENT.insert(table_name, obj_feat.features)
    print('insert vec id:', vec_id)
    res = MILVUS_CLIENT.search_vectors(table_name, obj_feat.features, top_k)
    # print('search res:', res)
    # take all ids and distances from results
    for item in res:
        print('item:', item)
        print('res:', item[0])
        print('id:', item[0].id)
        print('distance:', item[0].distance)


def test_search_vectors():
    vit_model = VitTiny224()
    obj_feat, candidate_box = vit_model.extract_primary_features('../data/test.jpg')
    print('candidate_box:', candidate_box)
    print('vector size:', len(obj_feat.features))
    res = MILVUS_CLIENT.search_vectors(DEFAULT_TABLE, obj_feat.features, 10)
    print(res)


def test_delete_collection():
    res = MILVUS_CLIENT.delete_collection(DEFAULT_TABLE)
    print(res)


def test_search_milvus_ops():
    table_name = 'test_collection'
    p_search_pre = (
        VitBase224().pipeline()
        .flat_map('vec', ('pred', 'distance'), search_milvus_ops(MILVUS_CLIENT, table_name, 5))
        .output('pred', 'distance')
    )

    res = p_search_pre('../data/objects.png')
    print(f'Number of search results: {res.size}')
    for i in range(res.size):
        it = res.get()
        print(f'{i}, search_res: {it[0]}, distance: {it[1]}')
