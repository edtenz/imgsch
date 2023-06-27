import hashlib
import os

import cv2
import numpy as np
import requests
from PIL import Image

from logger import LOGGER


def gen_file_key(file_path: str) -> str:
    """
    Generate MD5 hash of file
    :param file_path: path to file
    :return: md5 hash of file
    """
    # get file suffix
    suffix = os.path.splitext(file_path)[1]
    # suffix to lower case
    suffix = suffix.lower()
    # generate md5 hash of file
    md5_hash = md5_file(file_path)
    if md5_hash == '':
        return ''
    # return file key
    return f'{md5_hash}{suffix}'


def md5_file(file_path: str) -> str:
    """
    Calculate MD5 hash of file
    :param file_path:
    :return:
    """
    try:
        with open(file_path, 'rb') as f:
            file_content = f.read()
            if file_content:
                md5_hash = hashlib.md5(file_content).hexdigest()
                return md5_hash
            else:
                print(f"File '{file_path}' is empty.")
                return ''
    except Exception as e:
        print(f"Error calculating MD5 hash of file '{file_path}': {str(e)}")
        return ''


def md5_content(content: bytes) -> str:
    """
    Calculate MD5 hash of content
    :param content: content to calculate MD5 hash
    :return: md5 hash of content
    """
    return hashlib.md5(content).hexdigest()


def get_images(path):
    pics = []
    for f in os.listdir(path):
        if ((f.endswith(extension) for extension in
             ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']) and not f.startswith('.DS_Store')):
            pics.append(os.path.join(path, f))
    return pics


def load_image(path: str):
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for f in os.listdir(path):
            if ((f.endswith(extension) for extension in
                 ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']) and not f.startswith('.DS_Store')):
                yield os.path.join(path, f)
    else:
        raise ValueError(f"Path '{path}' is not a valid file or directory.")


def resize_image(image_path: str, width: int, height: int, output_dir: str, quality=70) -> str:
    """
    Resize the image to width and height and save it to the output_dir
    Args:
        image_path: image path
        width: new width
        height: new height
        output_dir: output directory
        quality: quality of the resized image, 1-100

    Returns: resized image path

    """
    image = Image.open(image_path)
    if width == 0 or height == 0:
        width, height = image.size

    resized_image = image.resize((width, height))

    # Create the 'resize' directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract the original image filename
    image_filename = os.path.basename(image_path)

    # Generate the output image path in the 'resize' directory
    output_path = os.path.join(output_dir, image_filename)

    resized_image.save(output_path, optimize=True, quality=quality)
    return output_path


def thumbnail(image_path: str, max_size: int, output_dir: str, quality=70) -> str:
    """
    Thumbnail the image to max_size and save it to the output_dir
    Args:
        image_path: image path
        max_size: max size of the image after resizing, width or height
        output_dir: path to the output directory
        quality: quality of the resized image, 1-100
    Returns: thumbnail image path
    """
    image = Image.open(image_path)
    image.thumbnail((max_size, max_size))

    # Create the 'resize' directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract the original image filename
    image_filename = os.path.basename(image_path)

    # Generate the output image path in the 'resize' directory
    output_path = os.path.join(output_dir, image_filename)

    image.save(output_path, optimize=True, quality=quality)
    return output_path


def thumbnail_ops(output_dir: str, max_size: int = 450, quality=60) -> callable:
    def wrapper(image_path: str) -> str:
        return thumbnail(image_path, max_size, output_dir, quality)

    return wrapper


def load_from_bytes(bs: bytes) -> np.ndarray:
    try:
        arr = np.asarray(bytearray(bs), dtype=np.uint8)
        return cv2.imdecode(arr, -1)
    except Exception as e:
        LOGGER.error('Load image from bytes failed, error msg: %s' % str(e))


def load_from_local(image_path: str) -> np.ndarray:
    return cv2.imread(image_path)


def load_from_remote(image_url: str) -> np.ndarray:
    try:
        r = requests.get(image_url, timeout=(20, 20))
        if r.status_code // 100 != 2:
            LOGGER.error('Download image from %s failed, error msg: %s, request code: %s ' % (image_url,
                                                                                              r.text,
                                                                                              r.status_code))
            return None
        return load_from_bytes(r.content)
    except Exception as e:
        LOGGER.error('Download image from %s failed, error msg: %s' % (image_url, str(e)))
        return False


def load_image_ops(image_path: str):
    bgr_cv_image = load_from_local(image_path)

    if bgr_cv_image is None:
        err = 'Read image %s failed' % image_path
        LOGGER.error('Read image %s failed' % image_path)
        raise RuntimeError(err)

    rgb_cv_image = cv2.cvtColor(bgr_cv_image, cv2.COLOR_BGR2RGB)
    return Image(rgb_cv_image, 'RGB')
