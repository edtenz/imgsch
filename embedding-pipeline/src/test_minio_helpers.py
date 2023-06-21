from minio_helpers import MINIO_CLIENT
from image_helper import md5_file


def test_upload_file():
    img_file = '../data/objects.png'
    image_key = md5_file(img_file)
    if MINIO_CLIENT.exists_object(image_key):
        print(f'object {image_key} exists')

    image_key = MINIO_CLIENT.upload_file(img_file)
    print(f'object {image_key} uploaded successfully')

    res = MINIO_CLIENT.download(image_key, '../objects-2.png')
    print(f'object {image_key} downloaded {res}')
