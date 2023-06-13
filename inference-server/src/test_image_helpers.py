import os

from image_helpers import resize_image, thumbnail


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
