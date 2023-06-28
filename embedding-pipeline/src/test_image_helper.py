from towhee import pipe

import image_helper


def test_gen_file_key():
    fkey = image_helper.gen_file_key('../data/objects.png')
    print(fkey)


def test_load_image():
    img_dir = '../data'
    for f in image_helper.load_image(img_dir):
        print(f)


def test_load_image_pipeline():
    img_dir = '../data'

    f_pipeline = (
        pipe.input('dir')
        .flat_map('dir', 'file', image_helper.load_image)
        .output('file')
    )

    res = f_pipeline(img_dir)
    size = res.size
    print(f'load {size} images')
    for i in range(size):
        print(res.get()[0])


def test_load_from_local():
    bs = image_helper.load_from_local('../data/objects.png')
    print(len(bs))
    assert len(bs) > 0
    'empty bytes'


def test_load_from_remote():
    bs = image_helper.load_from_remote('http://localhost:10086/api/imgsch/000c9b3463d25d5fee7bcb4c473393f3.jpg')
    print(len(bs))
    assert len(bs) > 0
    'empty bytes'


def test_load_image_ops():
    bs = image_helper.load_image_ops('../data/objects.png')
    print(len(bs))
    assert len(bs) > 0
    'empty bytes'


def test_load_http_image_ops():
    bs = image_helper.load_image_ops('http://localhost:10086/api/imgsch/000c9b3463d25d5fee7bcb4c473393f3.jpg')
    print(len(bs))
    print(bs)
    assert len(bs) > 0
    'empty bytes'
