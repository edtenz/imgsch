import threading

from grpcserver import start_grpc_server
from httpserver import start_http_server


def http_serve():
    print('Start http service...')
    start_http_server()


def grpc_serve():
    print('Start grpc service...')
    grpc_thread = threading.Thread(target=start_grpc_server, args=())
    grpc_thread.start()


if __name__ == '__main__':
    grpc_serve()
    http_serve()
