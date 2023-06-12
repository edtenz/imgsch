from extract import EXTRACTOR


def test_detect():
    key = '224d11f6b5d17a73c4d03546b433410a'
    box = (448, 153, 663, 375)
    res = EXTRACTOR.extract(key, box)
    print(res)
    print(len(res))


def test_detect_without_box():
    key = 'bdd2ebce0a9e44233099a469f4872e2c'
    box = None
    res = EXTRACTOR.extract(key, box)
    print(res)
    print(len(res))
