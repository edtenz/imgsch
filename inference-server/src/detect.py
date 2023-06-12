from pydantic import BaseModel
from towhee import pipe, ops, AutoConfig

from logs import LOGGER
from minio_helpers import download_object, remove_local_object


class BoundingBox(BaseModel):
    box: list[int]
    score: float
    label: str

    def to_dict(self) -> dict:
        """
        Convert BoundingBox to dict
        :return:  dict
        """
        return {
            'box': self.box,
            'score': self.score,
            'label': self.label
        }


class Detector(object):
    def __init__(self):
        self.auto_config = AutoConfig.LocalCPUConfig()
        self.obj_pipeline = (
            pipe.input('url')
            .map('url', 'img', ops.image_decode.cv2_rgb(), self.auto_config.config)  # decode image
            .flat_map('img', ('box', 'class', 'score'), ops.object_detection.yolo(),
                      self.auto_config.config)  # detect object
            .output('url', 'box', 'class', 'score')  # output
        )

    def detect_file(self, image_file: str) -> list[BoundingBox]:
        """
        Detect object from local file or url
        :param image_file: is the local file path
        :return: list of BoundingBox
        """

        bboxes = []
        res = self.obj_pipeline(image_file)
        if res.size == 0:
            return bboxes

        for i in range(res.size):
            item = res.get()
            # print('{}: url: {}, box: {}, class: {}, score: {}'.format(i, item[0], item[1], item[2], item[3]))
            bbox = BoundingBox(box=item[1], score=item[3], label=item[2])
            bboxes.append(bbox)
        return bboxes

    def detect(self, key: str) -> list[BoundingBox]:
        """
        Detect object from minio object
        :param key: is the object key in minio
        :return: list of BoundingBox
        """
        bboxes = []
        download_path = download_object(key)
        if download_path == '':
            LOGGER.error('download object failed, key: {}'.format(key))
            return bboxes

        bboxes = self.detect_file(download_path)
        LOGGER.debug('detect file: {}'.format(bboxes))
        remove_local_object(key)
        return bboxes


DETECTOR = Detector()
