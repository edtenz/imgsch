from towhee import pipe

from image_helper import gen_file_key, load_image


def test_gen_file_key():
    fkey = gen_file_key('../data/objects.png')
    print(fkey)


def test_load_image():
    img_dir = '../data'
    for f in load_image(img_dir):
        print(f)


def test_load_image_pipeline():
    img_dir = '../data'

    f_pipeline = (
        pipe.input('dir')
        .flat_map('dir', 'file', load_image)
        .output('file')
    )

    res = f_pipeline(img_dir)
    size = res.size
    print(f'load {size} images')
    for i in range(size):
        print(res.get()[0])
