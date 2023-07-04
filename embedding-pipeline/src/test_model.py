import image_helper
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


def test_vit224_url_extract_features():
    model = VitTiny224()
    obj_features = model.extract_features('http://localhost:10086/file/imgsch/000c9b3463d25d5fee7bcb4c473393f3.jpg')
    for obj_feat in obj_features:
        print(obj_feat)
        assert len(obj_feat.features) == 192


def test_vitTiny224_extract_primary_features():
    model = VitTiny224()
    obj_feat, candidate_box = model.extract_primary_features('../data/objects.png')
    print(obj_feat)
    print(candidate_box)
    assert len(obj_feat.features) == 192


def test_vitBase224_extract_primary_features():
    model = VitBase224()
    obj_feat, candidate_box = model.extract_primary_features('../data/objects.png')
    print(obj_feat)
    print(candidate_box)
    assert len(obj_feat.features) == 768


def test_vitBase224_url_extract_primary_features():
    model = VitBase224()
    obj_feat, candidate_box = model.extract_primary_features(
        'http://localhost:10086/file/imgsch/000c9b3463d25d5fee7bcb4c473393f3.jpg')
    print(obj_feat)
    print(candidate_box)
    assert len(obj_feat.features) == 768


def test_width_height():
    w, h = image_helper.get_image_dimensions('../data/objects.png')
    print(f'w: {w}, h: {h}')


def test_vitBase224_local_extract_primary_features():
    model = VitBase224()
    obj_feat, candidate_box = model.extract_primary_features('../data/objects.png')
    print(obj_feat)
    print(candidate_box)
    assert len(obj_feat.features) == 768


def test_pipeline():
    model = VitTiny224()
    f_pipeline = (
        model.pipeline()
        .output('box', 'label', 'score', 'vec')
    )
    res = f_pipeline('../data/objects.png')
    size = res.size
    print(f'load {size} images')
    for i in range(size):
        it = res.get()
        print(f'box: {it[0]}, label: {it[1]}, score: {it[2]}, vec: {it[3]}')
