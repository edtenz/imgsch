import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from config import HTTP_PORT
from detect import DETECTOR
from extract import EXTRACTOR
from logs import LOGGER
from minio_helpers import MINIO_CLIENT
from operations import do_detect, do_extract

app = FastAPI()


@app.get("/ping")
def ping():
    LOGGER.debug("ping")
    return {"message": "pong"}


@app.get('/detect')
def detect_img(key: str):
    try:
        LOGGER.debug(f"detect image: {key}")
        bboxes = [item.to_dict() for item in do_detect(key, MINIO_CLIENT, DETECTOR)]
        LOGGER.info(f"Successfully detect image: {key}, bboxes: {bboxes}")
        return JSONResponse({'status': True, 'msg': 'success', 'data': bboxes})
    except Exception as e:
        LOGGER.error(f"Get image error: {e}")
        return {'status': False, 'msg': e}, 400


@app.get('/extract')
def extract_imag(key: str, bbox: str):
    try:
        LOGGER.debug(f"extract image: {key} with bbox: {bbox}")
        # split bbox by comma
        box = get_bbox(bbox)
        features = do_extract(key, MINIO_CLIENT, EXTRACTOR, box)

        LOGGER.info(f"Successfully extract image: {key}, feature: {features[:5]}, length: {len(features)}")
        return JSONResponse({'status': True, 'msg': 'success', 'data': features.tolist()})
    except Exception as e:
        LOGGER.error(f"Get image error: {e}")
        return {'status': False, 'msg': e}, 400


def get_bbox(bbox: str) -> tuple[int, int, int, int]:
    """
    Get bbox from string
    :param bbox:  bbox string, like: '1,2,3,4'
    :return: box tuple, like: (1,2,3,4)
    """
    if bbox is None:
        return None
    box = [int(item) for item in bbox.split(',')]
    if len(box) != 4:
        return None
    return box[0], box[1], box[2], box[3]


def start_http_server():
    LOGGER.info(f"http listen on: {HTTP_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=int(HTTP_PORT))
