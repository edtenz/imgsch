from src.model import Resnet50


def test_extract_features():
    model = Resnet50()
    features = model.extract_features('../data/objects.png')
    print(features)
    assert len(features) == 2048
