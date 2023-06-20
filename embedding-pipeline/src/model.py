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
        self.feature_pipeline = (
            pipe.input('url')
            .map('url', 'img', ops.image_decode.cv2_rgb())
            .map('url', 'key', image_helper.md5_file)  # decode image
            .flat_map('img', ('box', 'label', 'score'), ops.object_detection.yolo())  # detect object
            .filter(('key', 'img', 'box', 'label', 'score'), ('key', 'img', 'box', 'label', 'score'),
                    'score', lambda x: x > 0.6)
            .flat_map(('img', 'box'), 'object', ops.towhee.image_crop())  # crop object
            .map('box', 'sbox', lambda x: ",".join(str(item) for item in x))  # box string
            .map('object', 'vec', ops.image_embedding.timm(model_name=model_name))  # extract feature
            .map('vec', 'vec', ops.towhee.np_normalize())
        )

    def pipeline(self):
        return self.feature_pipeline

    def extract_features(self, url: str) -> list[ObjectFeature]:
        """
        Extract feature from local file or url
        :param url: url or local file path
        :return: object features
        """
        p = (
            self.pipeline()
            .output('key', 'box', 'label', 'score', 'vec')
        )
        obj_feat_list = []
        res = p(url)
        if res.size == 0:
            return obj_feat_list

        for i in range(res.size):
            it = res.get()
            obj_feat = ObjectFeature(url=url,
                                     key=it[0],
                                     box=tuple(it[1]),
                                     label=it[2],
                                     score=it[3],
                                     features=it[4].tolist())
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
            .output('key', 'box', 'label', 'score', 'vec')
        )
        res = p(url)
        if res.size == 0:
            return None

        it = res.get()
        return ObjectFeature(url=url,
                             key=it[0],
                             box=tuple(it[1]),
                             label=it[2],
                             score=it[3],
                             features=it[4].tolist())


class Resnet50(Model):

    def __init__(self):
        super().__init__('resnet50')


class Vit224(Model):

    def __init__(self):
        super().__init__('vit_tiny_patch16_224')
