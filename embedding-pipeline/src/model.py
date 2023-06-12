from towhee import pipe, ops, AutoConfig


class Resnet50(object):

    def __init__(self):
        self.auto_config = AutoConfig.LocalCPUConfig()
        self.model_name = 'resnet50'
        self.feature_pipeline = (
            pipe.input('url')
            .map('url', 'img', ops.image_decode.cv2_rgb())  # decode image
            .flat_map('img', ('box', 'class', 'score'), ops.object_detection.yolo())  # detect object
            .flat_map(('img', 'box'), 'object', ops.towhee.image_crop())  # crop object
            .map('object', 'embedding', ops.image_embedding.timm(model_name=self.model_name))  # extract feature
            .map('embedding', 'embedding', ops.towhee.np_normalize())
            .output('embedding')  # output
        )

    def extract_features(self, url) -> list[float]:
        res = self.feature_pipeline(url)
        if res.size == 0:
            return []
        # take the first object and features
        return res.get()[0]
