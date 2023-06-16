import sys

sys.path.append("../src")
from detect import DETECTOR


def test_detect_file():
    img = '../data/objects.png'
    res = DETECTOR.detect_file(img)
    for item in res:
        print(item)
    assert len(res) == 2


def test_detect():
    key = '224d11f6b5d17a73c4d03546b433410a'
    res = DETECTOR.detect(key)
    for item in res:
        print(item)
    assert len(res) == 2
