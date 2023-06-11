from concurrent import futures

import grpc

import api_pb2
import api_pb2_grpc
from config import GRPC_PORT
from detect import detector
from extract import extractor
from logs import LOGGER


class FeatureService(api_pb2_grpc.FeatureServiceServicer):
    def Detect(self, request, context):
        LOGGER.info(f"Received Detect request: {request}")
        bboxes = detector.detect(request.key)
        if bboxes:
            header = api_pb2.ResponseHeader(code=0, msg="")
            boxes = []
            for bbox in bboxes:
                box = api_pb2.BoundingBox(box=bbox.box, score=bbox.score, label=bbox.cat)
                boxes.append(box)
            response = api_pb2.DetectResponse(header=header, bboxes=boxes)
            LOGGER.info(f"Detect response: {response}")
            return response
        else:
            header = api_pb2.ResponseHeader(code=1, msg="detect failed")
            LOGGER.warn(f"Detect error with response: {header}")
            return api_pb2.DetectResponse(header=header)

    def Extract(self, request, context):
        LOGGER.info(f"Received Extract request: {request}")
        box = get_box(request.bbox)
        features = extractor.extract(request.key, box)
        if features:
            header = api_pb2.ResponseHeader(code=0, msg="")
            response = api_pb2.ExtractResponse(header=header, features=features)
            LOGGER.info(f"Extract response: {response}")
            return response
        else:
            header = api_pb2.ResponseHeader(code=2, msg="extract failed")
            LOGGER.warn(f"Extract error with response: {header}")
            return api_pb2.ExtractResponse(header=header)


def get_box(bbox: api_pb2.BoundingBox) -> tuple[int, int, int, int]:
    """
    Get bbox from protobuf
    :param bbox: bbox protobuf
    :return: box tuple, like: (1,2,3,4)
    """
    if bbox is None:
        return None
    if len(bbox.box) != 4:
        return None
    return bbox.box[0], bbox.box[1], bbox.box[2], bbox.box[3]


def start_grpc_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    api_pb2_grpc.add_FeatureServiceServicer_to_server(FeatureService(), server)
    print(f"Starting gRPC server on :{GRPC_PORT}")
    port = f'[::]:{GRPC_PORT}'
    server.add_insecure_port(port)
    server.start()
    print("gRPC server started...")
    server.wait_for_termination()
