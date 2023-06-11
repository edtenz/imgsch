from detect import detect, obj_embedding


def test_detect():
    img = '../data/objects.png'
    res = detect(img)
    for item in res:
        print(item)
    assert len(res) == 2


def test_pipeline():
    # data = 'https://towhee.io/object-detection/yolo/raw/branch/main/objects.png'
    img = '../data/objects.png'
    res = obj_embedding(img)
    print(res.size)  # return 2
    for i in range(res.size):
        item = res.get()
        # print('res[{}]:'.format(i), item)
        print('{}: url: {}, box: {}, class: {}, score: {}'.format(i, item[0], item[1], item[2], item[3]))
        print('============')
