from image_helper import gen_file_key


def test_gen_file_key():
    fkey = gen_file_key('../data/objects.png')
    print(fkey)
