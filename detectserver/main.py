from towhee import pipe, ops, AutoConfig

auto_config = AutoConfig.LocalCPUConfig()

obj_embedding = (
    pipe.input('url')
    .map('url', 'img', ops.image_decode.cv2_rgb(), auto_config.config)  # decode image
    .flat_map('img', ('box', 'class', 'score'), ops.object_detection.yolo(), auto_config.config)  # detect object
    .output('url', 'box', 'class', 'score')  # output
)


class BoundingBox(object):
    def __int__(self):
        self.box = []
        self.score = 0.0
        self.cat = ''

    def __str__(self):
        return 'box: {}, score: {}, cat: {}'.format(self.box, self.score, self.cat)


def detect(url):
    bboxes = []
    res = obj_embedding(url)
    if res.size == 0:
        return bboxes

    for i in range(res.size):
        item = res.get()
        # print('{}: url: {}, box: {}, class: {}, score: {}'.format(i, item[0], item[1], item[2], item[3]))
        bbox = BoundingBox()
        bbox.box = item[1]
        bbox.cat = item[2]
        bbox.score = item[3]
        bboxes.append(bbox)
    return bboxes


if __name__ == '__main__':
    print('Start image detect service...')
    test_pipeline()
