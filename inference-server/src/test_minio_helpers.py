from minio_helpers import MINIO_CLIENT, calculate_md5, download_object, md5_hash


def test_upload():
    img_path = '../data/objects.png'
    object_name = calculate_md5(img_path)
    ok = MINIO_CLIENT.upload(object_name, img_path)
    assert ok
    "Upload failed"


def test_download():
    img_path = '../data/objects.png'
    download_path = '../data/objects_downloaded.png'
    object_name = calculate_md5(img_path)
    print(object_name)
    ok = MINIO_CLIENT.download(object_name, download_path)
    assert ok
    "Download failed"


def test_download2():
    object_name = '224d11f6b5d17a73c4d03546b433410a'
    download_path = download_object(object_name)
    assert download_path != '' "Download failed"


def test_md5_hash():
    content = b'Hello World'
    md5 = 'b10a8db164e0754105b7a99be72e3fe5'
    assert md5_hash(content) == md5
    "MD5 hash failed"
