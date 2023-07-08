from towhee import pipe

from image_helper import load_image, thumbnail_ops, gen_file_key
from minio_helpers import download_minio_ops, MinioClient, upload_minio_ops

minio_cli = MinioClient()


def test_upload_file():
    img_file = '../data/objects.png'
    image_key = gen_file_key(img_file)
    if minio_cli.exists_object(image_key):
        print(f'object {image_key} exists')

    image_key = minio_cli.upload_file(img_file)
    print(f'object {image_key} uploaded successfully')

    res = minio_cli.download(image_key, '../objects-2.png')
    print(f'object {image_key} downloaded {res}')


def test_download_minio_ops():
    key = '9880b8dd5e520f50c437be21372440f5'
    d = download_minio_ops(minio_cli)
    img_path = d(key)
    print(f'object {key} downloaded to {img_path}')


def test_load_path():
    img_dir = '/Users/edtenz/Downloads/JPEGImages'
    tmp_dir = '../tmp'
    bucket_name = 'imgsch'
    minio_cli = MinioClient(
        endpoint='localhost:9090',
        access_key='minioadmin',
        secret_key='minioadmin',
        secure=False,
        bucket_name=bucket_name,
    )
    f_pipeline = (
        pipe.input('dir')
        .flat_map('dir', 'file', load_image)
        .map('file', 'thumbnail_file', thumbnail_ops(tmp_dir, 450, 70))
        .map('thumbnail_file', 'thumbnail_key', gen_file_key)
        .map(('thumbnail_key', 'thumbnail_file'), 'upload_res', upload_minio_ops(minio_cli, bucket_name))
        .output('thumbnail_file', 'thumbnail_key', 'upload_res')
    )

    res = f_pipeline(img_dir)
    size = res.size
    print(f'load {size} images')
    for i in range(size):
        it = res.get()
        print(f'file: {it[0]}, key: {it[1]}, upload_res: {it[2]}')
