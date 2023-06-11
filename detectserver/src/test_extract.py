from extract import extractor


def test_detect():
    key = '224d11f6b5d17a73c4d03546b433410a'
    box = (448, 153, 663, 375)
    res = extractor.extract(key, box)
    print(res)
    print(len(res))


def test_detect_without_box():
    key = '224d11f6b5d17a73c4d03546b433410a'
    box = None
    res = extractor.extract(key, box)
    print(res)
    print(len(res))
