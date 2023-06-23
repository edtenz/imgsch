from image_helper import gen_file_key
from minio_helpers import MINIO_CLIENT, download_minio_ops


def test_upload_file():
    img_file = '../data/objects.png'
    image_key = gen_file_key(img_file)
    if MINIO_CLIENT.exists_object(image_key):
        print(f'object {image_key} exists')

    image_key = MINIO_CLIENT.upload_file(img_file)
    print(f'object {image_key} uploaded successfully')

    res = MINIO_CLIENT.download(image_key, '../objects-2.png')
    print(f'object {image_key} downloaded {res}')


def test_download_minio_ops():
    key = '9880b8dd5e520f50c437be21372440f5'
    d = download_minio_ops(MINIO_CLIENT)
    img_path = d(key)
    print(f'object {key} downloaded to {img_path}')
