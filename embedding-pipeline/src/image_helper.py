import csv
import hashlib
import os
from glob import glob
from pathlib import Path


def md5_file(file_path: str) -> str:
    """
    Calculate MD5 hash of file
    :param file_path: path to file
    :return: md5 hash of file
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
    for f in os.listdir(path):
        if ((f.endswith(extension) for extension in
             ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']) and not f.startswith('.DS_Store')):
            yield os.path.join(path, f)
