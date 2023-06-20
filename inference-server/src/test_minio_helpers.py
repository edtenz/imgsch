from image_helpers import md5_file
from minio_helpers import MINIO_CLIENT, download_object


def test_upload():
    img_path = '../data/objects.png'
    object_name = md5_file(img_path)
    ok = MINIO_CLIENT.upload(object_name, img_path)
    assert ok
    "Upload failed"


def test_download():
    img_path = '../data/objects.png'
    download_path = '../data/tmp/objects_downloaded.png'
    object_name = md5_file(img_path)
    print(object_name)
    ok = MINIO_CLIENT.download(object_name, download_path)
    assert ok
    "Download failed"


def test_download2():
    object_name = '224d11f6b5d17a73c4d03546b433410a'
    download_path = download_object(object_name, MINIO_CLIENT)
    assert download_path != '' "Download failed"
