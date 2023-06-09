from towhee import pipe, ops


def double_x(x):
    return x * 2


def simple_pipeline():
    print('Start simple pipeline...')
    # create pipeline
    p = (
        pipe.input('x')  # input x
        .map('x', 'y', lambda x: x + 1)  # add 1 to x
        .map('y', 'y', double_x)  # double y
        .output('y')  # output y
    )

    # run pipeline with input 10
    res = p(10).get()
    print(res)


def filter_pipeline():
    print('Start filter pipeline...')
    obj_filter_embedding = (
        pipe.input('url')
        .map('url', 'img', ops.image_decode.cv2_rgb())
        .map('img', 'obj_res', ops.object_detection.yolo())
        .filter(('img', 'obj_res'), ('img', 'obj_res'), 'obj_res', lambda x: len(x) > 0)
        .flat_map('obj_res', ('box', 'class', 'score'), lambda x: x)
        .flat_map(('img', 'box'), 'object', ops.towhee.image_crop())
        .map('object', 'embedding', ops.image_embedding.timm(model_name='resnet50'))
        .output('url', 'object', 'class', 'score', 'embedding')
    )

    data = ['https://towhee.io/object-detection/yolo/raw/branch/main/objects.png',
            'https://github.com/towhee-io/towhee/raw/main/assets/towhee_logo_square.png']
    res = obj_filter_embedding.batch(data)
    print(res)
    print(len(res))
    for item in res:
        print(item.get())
        print('================')


def img_pipeline():
    print('Start image pipeline...')
    img_embedding = (
        pipe.input('url')
        .map('url', 'img', ops.image_decode.cv2())
        .map('img', 'embedding', ops.image_embedding.timm(model_name='resnet50'))
        .output('embedding')
    )

    url = 'https://github.com/towhee-io/towhee/raw/main/towhee_logo.png'
    res = img_embedding(url).get()
    print(res)
    print(res[0])
    print(len(res[0]))


def img_extract_pipeline():
    print('Start image feature pipeline...')
    img_embedding = (
        pipe.input('img_file')
        .map('img_file', 'img', ops.image_decode.cv2())
        .map('img', 'embedding', ops.image_embedding.timm(model_name='resnet50'))
        .map('embedding', 'vec', ops.towhee.np_normalize())
        .output('vec')
    )

    img_file = 'test.jpg'
    res = img_embedding(img_file).get()
    print(res)
    print(res[0])
    print(len(res[0]))


def img_detect_pipeline():
    print('Start image detect pipeline...')
    obj_embedding = (
        pipe.input('url')
        .map('url', 'img', ops.image_decode.cv2_rgb())  # decode image
        .flat_map('img', ('box', 'class', 'score'), ops.object_detection.yolo())  # detect object
        .flat_map(('img', 'box'), 'object', ops.towhee.image_crop())  # crop object
        .map('object', 'embedding', ops.image_embedding.timm(model_name='resnet50'))  # extract feature
        .output('url', 'box', 'class', 'score', 'object', 'embedding')  # output
    )

    # data = 'https://towhee.io/object-detection/yolo/raw/branch/main/objects.png'
    data = 'test.jpg'
    res = obj_embedding(data)
    print('res:', res)
    print(res.size)  # return 2
    for i in range(res.size):
        print('res[{}]:'.format(i), res.get())
        print('============')


def img_text_pipeline():
    in_pipe = pipe.input('url', 'text')

    img_embedding = (
        in_pipe.map('url', 'img', ops.image_decode.cv2_rgb())
        .map('img', 'img_embedding',
             ops.image_text_embedding.clip(model_name='clip_vit_base_patch16', modality='image'))
    )

    text_embedding = in_pipe.map('text', 'text_embedding',
                                 ops.image_text_embedding.clip(model_name='clip_vit_base_patch16', modality='text'))

    img_text_embedding = (
        img_embedding.concat(text_embedding)
        .output('img', 'text', 'img_embedding', 'text_embedding')
    )

    img = 'https://towhee.io/object-detection/yolov5/raw/branch/main/test.png'
    text = 'A dog looking at a computer in bed.'
    res = img_text_embedding(img, text)
    print(res.get())


if __name__ == '__main__':
    simple_pipeline()
    filter_pipeline()
    # img_pipeline()
    img_extract_pipeline()
    img_detect_pipeline()
    img_text_pipeline()
