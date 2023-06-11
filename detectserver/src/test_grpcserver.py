import grpc
from google.protobuf import json_format

import api_pb2
import api_pb2_grpc
from config import GRPC_PORT


def test_detect():
    with grpc.insecure_channel(f'localhost:{GRPC_PORT}') as channel:
        stub = api_pb2_grpc.FeatureServiceStub(channel)
        header = api_pb2.RequestHeader(request_id='abc123')
        response = stub.Detect(api_pb2.DetectRequest(header=header, key='224d11f6b5d17a73c4d03546b433410a'))
        # print(response)
        print(json_format.MessageToJson(response))
        assert response.header.code == 0
        assert len(response.bboxes) == 2


def test_extract():
    with grpc.insecure_channel(f'localhost:{GRPC_PORT}') as channel:
        stub = api_pb2_grpc.FeatureServiceStub(channel)
        header = api_pb2.RequestHeader(request_id='abc123')
        bbox = api_pb2.BoundingBox(box=[448, 153, 663, 375])
        response = stub.Extract(
            api_pb2.ExtractRequest(header=header, key='224d11f6b5d17a73c4d03546b433410a', bbox=bbox))
        # print(response)
        print(json_format.MessageToJson(response))
        assert response.header.code == 0
