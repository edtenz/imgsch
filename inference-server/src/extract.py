from towhee import pipe, ops, AutoConfig

from image_helpers import get_image_dimensions
from logs import LOGGER


class Extractor(object):

    def __init__(self):
        self.auto_config = AutoConfig.LocalCPUConfig()
        # timm.list_models(pretrained=True)
        # supported models list: https://towhee.io/image-embedding/timm
        # self.model_name = 'resnet50'
        self.model_name = 'vit_tiny_patch16_224'
        self.feature_pipeline = (
            pipe.input('url', 'box')
            .map('url', 'img', ops.image_decode.cv2_rgb())  # decode image
            .flat_map(('img', 'box'), 'object', ops.towhee.image_crop())  # crop object
            .map('object', 'embedding', ops.image_embedding.timm(model_name=self.model_name))  # extract feature
            .map('embedding', 'embedding', ops.towhee.np_normalize())  # normalize feature
            .output('url', 'embedding')  # output
        )

    def extract(self, url: str, box: tuple[int, int, int, int]) -> list[float]:
        """
        Extract feature from local file or url
        :param url: is url or local file path
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


EXTRACTOR = Extractor()
