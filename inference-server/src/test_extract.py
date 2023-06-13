from towhee import ops

from extract import EXTRACTOR


def test_list_models():
    print('Test list models:')
    op = ops.image_embedding.timm().get_op()
    full_list = op.supported_model_names()
    print(f'Full Models: {len(full_list)}')
    print(full_list)
    onnx_list = op.supported_model_names(format='onnx')
    print(f'Onnx-support/Total Models: {len(onnx_list)}/{len(full_list)}')


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
