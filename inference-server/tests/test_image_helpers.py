import os

from image_helpers import resize_image, thumbnail, md5_content


def test_resize_image():
    image_path = '../data/objects.png'
    max_size = 224
    output_dir = '../data/resize'
    resize_image(image_path, max_size, max_size, output_dir, quality=70)
    assert os.path.exists(output_dir)
    'resize directory does not exist'


def test_thumbnail():
    image_path = '../data/objects.png'
    max_size = 224
    output_dir = '../data/resize'
    thumbnail(image_path, max_size, output_dir, quality=50)
    assert os.path.exists(output_dir)
    'thumbnail directory does not exist'


def test_md5_hash():
    content = b'Hello World'
    md5 = 'b10a8db164e0754105b7a99be72e3fe5'
    assert md5_content(content) == md5
    "MD5 hash failed"
