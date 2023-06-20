from src.model import Resnet50, Vit224


def test_resnet50_extract_features():
    model = Resnet50()
    features = model.extract_features('../data/objects.png')
    print(features)
    assert len(features) == 2048


def test_vit224_extract_features():
    model = Vit224()
    features = model.extract_features('../data/objects.png')
    print(features)
    assert len(features) == 192
