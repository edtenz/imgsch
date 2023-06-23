import hashlib
import os


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
