from concurrent import futures

import grpc

import api_pb2 as pb2
import api_pb2_grpc as pb2_grpc
from config import GRPC_PORT
from detect import DETECTOR
from extract import EXTRACTOR
from logger import LOGGER
from minio_helpers import MINIO_CLIENT
from operations import do_detect, do_extract


class FeatureService(pb2_grpc.FeatureServiceServicer):
    def Detect(self, request, context):
        LOGGER.debug(f"Received Detect request: {request}")
        bboxes = do_detect(request.key, MINIO_CLIENT, DETECTOR)
        if bboxes:
            header = pb2.ResponseHeader(code=0, msg="")
            boxes = []
            for bbox in bboxes:
                box = pb2.BoundingBox(box=bbox.box, score=bbox.score, label=bbox.label)
                boxes.append(box)
            response = pb2.DetectResponse(header=header, bboxes=boxes)
            LOGGER.debug(f"Detect response: {response}")
            return response
        else:
            header = pb2.ResponseHeader(code=1, msg="detect failed")
            LOGGER.warn(f"Detect error with response: {header}")
            return pb2.DetectResponse(header=header)

    def Extract(self, request, context):
        LOGGER.debug(f"Received Extract request: {request}")
        box = get_box(request.box)
        features = do_extract(request.key, MINIO_CLIENT, EXTRACTOR, box)
        if features is None or len(features) == 0:
            header = pb2.ResponseHeader(code=2, msg="extract failed")
            LOGGER.warn(f"Extract error with response: {header}")
            return pb2.ExtractResponse(header=header)
        else:
            header = pb2.ResponseHeader(code=0, msg="")
            response = pb2.ExtractResponse(header=header, features=features)
            LOGGER.debug(f"Extract response: {response}")
            return response


def get_box(box) -> tuple[int, int, int, int]:
    """
    Get bbox from protobuf
    :param box: box array, like: [1,2,3,4]
    :return: box tuple, like: (1,2,3,4)
    """
    if box is None:
        return None
    if len(box) != 4:
        return None
    return box[0], box[1], box[2], box[3]


def start_grpc_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_FeatureServiceServicer_to_server(FeatureService(), server)
    print(f"Starting gRPC server on :{GRPC_PORT}")
    port = f'[::]:{GRPC_PORT}'
    server.add_insecure_port(port)
    server.start()
    print("gRPC server started...")
    server.wait_for_termination()
