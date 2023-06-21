from pydantic import BaseModel
from towhee import pipe, ops, AutoConfig

import image_helper


class ObjectFeature(BaseModel):
    url: str
    key: str
    box: tuple[int, int, int, int]
    label: str
    score: float
    features: list[float]

    def to_dict(self) -> dict:
        """
        Convert ObjectFeature to dict
        :return:  dict
        """
        return {
            'url': self.url,
            'key': self.key,
            'box': self.box,
            'label': self.label,
            'score': self.score,
            'features': self.features
        }

    def __str__(self):
        return self.to_dict().__str__()


class Model(object):

    def __init__(self, model_name: str):
        self.auto_config = AutoConfig.LocalCPUConfig()
        self.detect_pipeline = (
            pipe.input('url')
            .filter(('url'), ('url'), 'url', lambda x: x is not None)  # filter invalid url
            .map('url', 'key', image_helper.md5_file)  # generate key
            .map('url', 'img', ops.image_decode.cv2_rgb())  # decode image
            .flat_map('img', ('box', 'label', 'score'), ops.object_detection.yolo())  # detect object
            .filter(('url', 'key', 'img', 'box', 'label', 'score'), ('url', 'key', 'img', 'box', 'label', 'score'),
                    'score', lambda x: x > 0.6)
        )
        self.extract_pipeline = (
            self.detect_pipeline
            .flat_map(('img', 'box'), 'object', ops.towhee.image_crop())  # crop object
            .map('box', 'sbox', lambda x: ",".join(str(item) for item in x))  # box string
            .map('object', 'vec', ops.image_embedding.timm(model_name=model_name))  # extract feature
            .map('vec', 'vec', ops.towhee.np_normalize())
        )

    def pipeline(self):
        """
        Get feature pipeline
        :return: pipeline: output('url', 'key', 'box', 'label', 'score', 'vec')
        """
        return self.extract_pipeline

    def extract_features(self, url: str) -> list[ObjectFeature]:
        """
        Extract feature from local file or url
        :param url: url or local file path
        :return: object features
        """
        p = (
            self.pipeline()
            .output('url', 'key', 'box', 'label', 'score', 'vec')
        )
        obj_feat_list = []
        res = p(url)
        if res.size == 0:
            return obj_feat_list

        for i in range(res.size):
            it = res.get()
            obj_feat = ObjectFeature(url=it[0],
                                     key=it[1],
                                     box=tuple(it[2]),
                                     label=it[3],
                                     score=it[4],
                                     features=it[5].tolist())
            # append to list
            obj_feat_list.append(obj_feat)
            # take the first 3 objects
            if i > 2:
                break
        return obj_feat_list

    def extract_primary_features(self, url: str) -> ObjectFeature:
        """
        Extract feature from local file or url
        :param url: url or local file path
        :return: object features
        """
        p = (
            self.pipeline()
            .output('url', 'key', 'box', 'label', 'score', 'vec')
        )
        res = p(url)
        if res.size == 0:
            return None

        it = res.get()
        return ObjectFeature(url=it[0],
                             key=it[1],
                             box=tuple(it[2]),
                             label=it[3],
                             score=it[4],
                             features=it[5].tolist())


class Resnet50(Model):

    def __init__(self):
        super().__init__('resnet50')


class VitTiny224(Model):

    def __init__(self):
        super().__init__('vit_tiny_patch16_224')


class VitBase224(Model):

    def __init__(self):
        super().__init__('vit_base_patch16_224')
