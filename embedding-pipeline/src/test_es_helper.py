from towhee import pipe, ops

from config import ES_HOST, ES_PORT
from model import VitBase224


def test_insert_doc_ops():
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
