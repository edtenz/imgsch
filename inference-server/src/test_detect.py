from detect import DETECTOR


def test_detect_file():
    img = '../data/objects.png'
    res = DETECTOR.detect(img)
    for item in res:
        print(item)
    assert len(res) == 2
