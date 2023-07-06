from towhee import pipe, ops

from config import ES_HOST, ES_PORT, ES_INDEX
from es_helpers import EsClient
from model import VitBase224


def test_insert_doc_ops_pipe():
    img_url = 'http://localhost:10086/file/imgsch/000c9b3463d25d5fee7bcb4c473393f3.jpg'
    model = VitBase224()
    obj_feat, candidate_box = model.extract_primary_features(img_url)
    print("obj_feat: ", obj_feat)

    # join the bbox to a string
    bbox = ','.join([str(x) for x in list(obj_feat.bbox.box)])

    example_doc = {
        'image_key': '000c9b3463d25d5fee7bcb4c473393f3',
        'image_url': img_url,
        'bbox': bbox,
        'bbox_score': obj_feat.bbox.score,
        'label': obj_feat.bbox.label,
        'features': obj_feat.features,
    }

    print("doc: ", example_doc)

    es_insert = (
        pipe.input('doc')
        .map('doc', 'res', ops.elasticsearch.index_client(
            host=ES_HOST, port=ES_PORT, index_name='imgsch'
        ))
        .output('doc', 'res')
    )

    res = es_insert(example_doc)  # OR: es_insert([example_doc])
    size = res.size
    if size == 0:
        print("no result")
        return
    for i in range(size):
        it = res.get()
        print(it)


def test_exist_index():
    es_cli = EsClient(host=ES_HOST, port=ES_PORT)
    existing = es_cli.exist_index(ES_INDEX)
    print("existing: ", existing)


def test_insert_doc():
    img_url = 'http://localhost:10086/file/imgsch/000c9b3463d25d5fee7bcb4c473393f3.jpg'
    model = VitBase224()
    obj_feat, candidate_box = model.extract_primary_features(img_url)
    print("obj_feat: ", obj_feat)

    # join the bbox to a string
    bbox = ','.join([str(x) for x in list(obj_feat.bbox.box)])

    doc = {
        'image_key': '000c9b3463d25d5fee7bcb4c473393f3',
        'image_url': img_url,
        'bbox': bbox,
        'bbox_score': obj_feat.bbox.score,
        'label': obj_feat.bbox.label,
        'features': obj_feat.features,
    }
    print("doc: ", doc)

    es_cli = EsClient(host=ES_HOST, port=ES_PORT)
    index_name = ES_INDEX
    ok = es_cli.insert_doc(index_name, doc, id=doc['image_key'])
    print("ok: ", ok)


def test_insert_batch():
    model = VitBase224()

    imgs = {
        '07dd1974a83862447b3dfa23957a4cfc': 'http://localhost:10086/file/imgsch/07dd1974a83862447b3dfa23957a4cfc.jpg',
        '026efe22c07322e7842a250ad1d3213b': 'http://localhost:10086/file/imgsch/026efe22c07322e7842a250ad1d3213b.jpg',
        '0592a3537dfe80e8e62523c7b6bbf58b': 'http://localhost:10086/file/imgsch/0592a3537dfe80e8e62523c7b6bbf58b.jpg',
    }

    docs = []
    for img_key, img_url in imgs.items():
        obj_feat, candidate_box = model.extract_primary_features(img_url)
        bbox = ','.join([str(x) for x in list(obj_feat.bbox.box)])
        doc = {
            'image_key': img_key,
            'image_url': img_url,
            'bbox': bbox,
            'bbox_score': obj_feat.bbox.score,
            'label': obj_feat.bbox.label,
            'features': obj_feat.features,
        }

        docs.append({"_id": img_key, "_source": doc})

    es_cli = EsClient(host=ES_HOST, port=ES_PORT)
    index_name = ES_INDEX
    res = es_cli.insert_batch(index_name, docs)
    print("res: ", res)


def test_create_index():
    es_cli = EsClient(host=ES_HOST, port=ES_PORT)
    index_name = 'testimgsch1'

    body = {
        "settings": {
            "index": {
                "refresh_interval": "180s",
                "number_of_replicas": "0"
            }
        },
        "mappings": {
            "properties": {
                "image_key": {
                    "type": "keyword",
                    "index": True
                },
                "image_url": {
                    "type": "text",
                    "index": False
                },
                "bbox": {
                    "type": "keyword",
                    "index": False
                },
                "bbox_score": {
                    "type": "float",
                    "index": True
                },
                "label": {
                    "type": "text",
                    "index": True
                },
                "features": {
                    "type": "dense_vector",
                    "dims": 768,
                    "index": True,
                    "similarity": "dot_product",
                    "index_options": {
                        "type": "hnsw",
                        "m": 16,
                        "ef_construction": 256
                    }
                }
            }
        }
    }

    ok = es_cli.create_index(index_name=index_name, body=body)
    print("ok: ", ok)


def test_query():
    es_cli = EsClient(host=ES_HOST, port=ES_PORT)

    index_name = ES_INDEX
    img_url = 'http://localhost:10086/file/imgsch/07dd1974a83862447b3dfa23957a4cfc.jpg'
    model = VitBase224()
    obj_feat, candidate_box = model.extract_primary_features(img_url)
    # print("obj_feat: ", obj_feat)
    query_vector = obj_feat.features

    knn_query = {
        "knn": {
            "field": "features",
            "query_vector": query_vector,
            "k": 10,
            "num_candidates": 10
        },
        "_source": {
            "excludes": ["features"]
        }
    }

    res = es_cli.query(index_name, knn_query)
    print("res: ", res)
