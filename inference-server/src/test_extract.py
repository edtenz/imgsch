from extract import EXTRACTOR


def test_extract():
    key = '224d11f6b5d17a73c4d03546b433410a'
    box = (448, 153, 663, 375)
    res = EXTRACTOR.extract(key, box)
    print(res)
    print(len(res))


def test_extract_without_box():
    key = '224d11f6b5d17a73c4d03546b433410a'
    box = None
    res = EXTRACTOR.extract(key, box)
    print(res)
    print(len(res))
