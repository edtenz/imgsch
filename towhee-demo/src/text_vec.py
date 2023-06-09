from towhee import AutoPipes


def start():
    p = AutoPipes.pipeline('sentence_embedding')
    output = p('Hello World.').get()
    print(output)
    # print output length
    print(len(output[0]))


if __name__ == '__main__':
    start()
    quit()
