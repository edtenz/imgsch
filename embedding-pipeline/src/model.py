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


class Resnet50(object):

    def __init__(self):
        self.auto_config = AutoConfig.LocalCPUConfig()
        self.model_name = 'resnet50'
        # self.model_name = 'vit_tiny_patch16_224'
        self.feature_pipeline = (
            pipe.input('url')
            .map('url', 'img', ops.image_decode.cv2_rgb())  # decode image
            .flat_map('img', ('box', 'class', 'score'), ops.object_detection.yolo())  # detect object
            .flat_map(('img', 'box'), 'object', ops.towhee.image_crop())  # crop object
            .map('object', 'embedding', ops.image_embedding.timm(model_name=self.model_name))  # extract feature
            .map('embedding', 'embedding', ops.towhee.np_normalize())
            .output('embedding')  # output
        )

    def extract_features(self, url: str) -> list[float]:
        """
        Extract feature from local file or url
        :param url: url or local file path
        :return: features
        """
        res = self.feature_pipeline(url)
        if res.size == 0:
            return []
        # take the first object and features
        return res.get()[0]


class Vit224(object):

    def __init__(self):
        self.auto_config = AutoConfig.LocalCPUConfig()
        self.model_name = 'vit_tiny_patch16_224'
        self.feature_pipeline = (
            pipe.input('url')
            .map('url', 'img', ops.image_decode.cv2_rgb())
            .map('url', 'key', image_helper.md5_file)  # decode image
            .flat_map('img', ('box', 'label', 'score'), ops.object_detection.yolo())  # detect object
            .flat_map(('img', 'box'), 'object', ops.towhee.image_crop())  # crop object
            .map('object', 'embedding', ops.image_embedding.timm(model_name=self.model_name))  # extract feature
            .map('embedding', 'embedding', ops.towhee.np_normalize())
            .output('key', 'box', 'label', 'score', 'embedding')  # output
        )

    def pipeline(self, url: str):
        return self.feature_pipeline(url)

    def extract_features(self, url: str) -> list[ObjectFeature]:
        """
        Extract feature from local file or url
        :param url: url or local file path
        :return: object features
        """
        obj_feat_list = []
        res = self.pipeline(url)
        if res.size == 0:
            return obj_feat_list

        for i in range(res.size):
            it = res.get()
            obj_feat = ObjectFeature(url=url, key=it[0], box=tuple(it[1]),
                                     label=it[2], score=it[3],
                                     features=it[4].tolist())
            # append to list
            obj_feat_list.append(obj_feat)
            # take the first 5 objects
            if i > 4:
                break
        return obj_feat_list
