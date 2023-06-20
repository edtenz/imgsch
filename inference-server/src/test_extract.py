from towhee import ops

from extract import EXTRACTOR
from src import extract


def test_list_models():
    print('Test list models:')
    op = ops.image_embedding.timm().get_op()
    full_list = op.supported_model_names()
    print(f'Full Models: {len(full_list)}')
    print(full_list)
    onnx_list = op.supported_model_names(format='onnx')
    print(f'Onnx-support/Total Models: {len(onnx_list)}/{len(full_list)}')


def test_extract():
    key = '../data/objects.png'
    box = (448, 153, 663, 375)
    res = extract.EXTRACTOR.extract(key, box)
    print(res)
    print(len(res))


def test_extract_without_box():
    key = '../data/objects.png'
    box = None
    res = EXTRACTOR.extract(key, box)
    print(res)
    print(len(res))
