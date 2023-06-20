from src.model import Resnet50, Vit224


def test_resnet50_extract_features():
    model = Resnet50()
    features = model.extract_features('../data/objects.png')
    print(features)
    assert len(features) == 2048


def test_vit224_extract_features():
    model = Vit224()
    obj_features = model.extract_features('../data/objects.png')
    for obj_feat in obj_features:
        print(obj_feat)
        assert len(obj_feat.features) == 192
    # assert len(features) == 192


def test_vit224_extract_features2():
    model = Vit224()
    res = model.pipeline('../data/objects.png')
    size = res.size
    print('size:', size)
    for i in range(size):
        it = res.get()
        print(it[0], it[1], it[2], it[3])
