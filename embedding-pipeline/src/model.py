from pydantic import BaseModel
from towhee import pipe, ops, AutoConfig

from logger import LOGGER


class BoundingBox(BaseModel):
    box: tuple[int, int, int, int]
    label: str
    score: float

    def __str__(self):
        return f"box: {self.box}, label: {self.label}, score: {self.score}"

    def to_dict(self):
        return {
            "box": list(self.box),
            "label": self.label,
            "score": self.score
        }


class ObjectFeature(BaseModel):
    url: str
    bbox: BoundingBox
    features: list[float]

    def to_dict(self) -> dict:
        """
        Convert ObjectFeature to dict
        :return:  dict
        """
        return {
            'url': self.url,
            'bbox': self.bbox.to_dict(),
            'features': self.features
        }

    def __str__(self):
        return self.to_dict().__str__()


class ImageFeatureModel(object):

    def __init__(self, model_name: str):
        self.auto_config = AutoConfig.LocalCPUConfig()
        self.model_name = model_name

        self.detect_pipeline = (
            pipe.input('url')
            .map('url', 'img', ops.image_decode.cv2_rgb())  # decode image
            .flat_map('img', ('box', 'label', 'score'), ops.object_detection.yolo())  # detect object
            .filter(('img', 'box', 'label', 'score'), ('img', 'box', 'label', 'score'),
                    'score', lambda x: x > 0.5)
        )  # detect pipeline for detect objects in image

        self.extract_pipeline = (
            self.detect_pipeline
            .flat_map(('img', 'box'), 'object', ops.towhee.image_crop())  # crop object
            .map('box', 'sbox', lambda x: ",".join(str(item) for item in x))  # box string
            .map('object', 'vec', ops.image_embedding.timm(model_name=model_name))  # extract feature
            .map('vec', 'vec', ops.towhee.np_normalize())
        )  # extract pipeline for extract features from objects

    def pipeline(self):
        """
        Get feature pipeline
        :return: pipeline: output('url', 'box', 'label', 'score', 'vec')
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
            .output('url', 'box', 'label', 'score', 'vec')
        )
        obj_feat_list = []
        res = p(url)
        if res.size == 0:
            return obj_feat_list

        for i in range(res.size):
            it = res.get()
            obj_feat = ObjectFeature(url=it[0],
                                     bbox=BoundingBox(box=tuple(it[1]),
                                                      label=it[2],
                                                      score=it[3]),
                                     features=it[4].tolist())
            # append to list
            obj_feat_list.append(obj_feat)
            # take the first 3 objects
            if i > 2:
                break
        return obj_feat_list

    def extract_primary_features(self, url: str) -> (ObjectFeature, list[BoundingBox]):
        """
        Extract feature from local file or url
        :param url: url or local file path
        :return: object features, candidate bbox list: (box, label, score)
        """

        decode_p = (
            pipe.input('url')
            .map('url', 'img', ops.image_decode.cv2_rgb())  # decode image
            .output('img')
        )

        decode_res = decode_p(url)
        if decode_res.size == 0:
            return None, []
        img = decode_res.get()[0]
        full_height, full_width, _ = img.shape
        detect_p = (
            pipe.input('img')
            .flat_map('img', ('box', 'label', 'score'), ops.object_detection.yolo())  # detect object
            .output('box', 'label', 'score')
        )

        detect_res = detect_p(url)
        box = [0, 0, 0 + full_width, 0 + full_height]
        label = ''
        score = 0.0
        candidate_list = []

        if detect_res.size == 0:
            LOGGER.debug(f'No object detected of {url}')
        else:
            for i in range(detect_res.size):
                detect_item = detect_res.get()
                if i == 0:
                    box, label, score = detect_item[0], detect_item[1], detect_item[2]
                else:
                    candidate_list.append(BoundingBox(box=detect_item[0],
                                                      label=detect_item[1],
                                                      score=detect_item[2]))

        extract_p = (
            pipe.input('img', 'box')
            .flat_map(('img', 'box'), 'object', ops.towhee.image_crop())  # crop object
            .map('box', 'sbox', lambda x: ",".join(str(item) for item in x))  # box string
            .map('object', 'vec', ops.image_embedding.timm(model_name=self.model_name))  # extract feature
            .map('vec', 'vec', ops.towhee.np_normalize())
            .output('vec')
        )

        extract_res = extract_p(img, box)
        bbox = BoundingBox(box=tuple(box), label=label, score=score)
        if extract_res.size == 0:
            return ObjectFeature(url=url, bbox=bbox, features=None), candidate_list

        extract_item = extract_res.get()
        return ObjectFeature(url=url, bbox=bbox, features=extract_item[0].tolist()), candidate_list


class Resnet50(ImageFeatureModel):

    def __init__(self):
        super().__init__('resnet50')


class VitTiny224(ImageFeatureModel):

    def __init__(self):
        super().__init__('vit_tiny_patch16_224')


class VitBase224(ImageFeatureModel):

    def __init__(self):
        super().__init__('vit_base_patch16_224')


def extract_features_ops(model: ImageFeatureModel) -> callable:
    """
    Extract feature from local file or url
    :param model: model
    :return: (sbox, label, score, features)
    """

    def wrapper(url: str) -> (str, str, float, list[float]):
        obj_feat, candidate_boxes = model.extract_primary_features(url)
        if obj_feat is None:
            return '', '', 0.0, []
        if obj_feat.bbox is None:
            return '', '', 0.0, []
        bbox = obj_feat.bbox
        sbox = ','.join(str(item) for item in list(bbox.box))
        return sbox, bbox.label, bbox.score, obj_feat.features

    return wrapper
