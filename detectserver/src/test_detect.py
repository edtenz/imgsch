from detect import Detector


def test_detect_file():
    img = '../data/objects.png'
    detector = Detector()
    res = detector.detect_file(img)
    for item in res:
        print(item)
    assert len(res) == 2


def test_detect():
    key = '224d11f6b5d17a73c4d03546b433410a'
    detector = Detector()
    res = detector.detect(key)
    for item in res:
        print(item)
    assert len(res) == 2
