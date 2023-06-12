from PIL import Image
from towhee import pipe, ops, AutoConfig

from logs import LOGGER
from minio_helpers import download_object, remove_local_object


class Extractor(object):

    def __init__(self):
        self.auto_config = AutoConfig.LocalCPUConfig()
        # self.model_name = 'resnet50'
        self.model_name = 'vit_large_patch16_224'
        self.feature_pipeline = (
            pipe.input('url', 'box')
            .map('url', 'img', ops.image_decode.cv2_rgb())  # decode image
            .flat_map(('img', 'box'), 'object', ops.towhee.image_crop())  # crop object
            .map('object', 'embedding', ops.image_embedding.timm(model_name=self.model_name))  # extract feature
            .map('embedding', 'embedding', ops.towhee.np_normalize())  # normalize feature
            .output('url', 'embedding')  # output
        )

    def extract_file(self, url: str, box: tuple[int, int, int, int]) -> list[float]:
        """
        Extract feature from local file or url
        :param url: url or local file path
        :param box: box of object
        :return: features
        """
        try:
            if box is None:
                LOGGER.info('box is None')
                w, h = get_image_dimensions(url)
                LOGGER.info('w: {}, h: {}'.format(w, h))
                box = (0, 0, w, h)

            res = self.feature_pipeline(url, box)
            if res.size == 0:
                LOGGER.warn('extract feature failed, url: {}, box: {}'.format(url, box))
                return []
            return res.get()[1]
        except Exception as e:
            LOGGER.error('extract feature failed, url: {}, box: {}, error: {}'.format(url, box, e))
            return []

    def extract(self, key: str, box: tuple[int, int, int, int]) -> list[float]:
        """
        Extract feature from minio object
        :param key: object name in minio
        :param box: box of object
        :return: features of object
        """
        download_path = download_object(key)
        if download_path == '':
            LOGGER.error('download object failed, key: {}'.format(key))
            return []
        features = self.extract_file(download_path, box)
        remove_local_object(key)
        return features


def get_image_dimensions(file_path) -> tuple[int, int]:
    """
    Get image dimensions
    :param file_path:  image file path
    :return:  image dimensions, width and height
    """
    with Image.open(file_path) as image:
        width, height = image.size
        return width, height


EXTRACTOR = Extractor()
