import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from config import PORT
from detect import detect_map_list, detect_map_list_batch
from logs import LOGGER

app = FastAPI()


@app.get("/ping")
def ping():
    LOGGER.debug("ping")
    return {"message": "pong"}


@app.get('/detect')
def detect_img(img: str):
    try:
        LOGGER.debug(f"detect image: {img}")
        bboxes = detect_map_list(img)
        LOGGER.info(f"Successfully detect image: {img}, bboxes: {bboxes}")
        return JSONResponse({'status': True, 'msg': 'success', 'data': bboxes})
    except Exception as e:
        LOGGER.error(f"Get image error: {e}")
        return {'status': False, 'msg': e}, 400


@app.post('/detect_batch')
def detect_img_batch(imgs: list):
    try:
        LOGGER.debug(f"detect images: {imgs}")
        boxes = detect_map_list_batch(imgs)
        return JSONResponse({'status': True, 'msg': 'success', 'data': boxes})
    except Exception as e:
        LOGGER.error(f"Get image error: {e}")
        return {'status': False, 'msg': e}, 400


def start_http_server():
    LOGGER.info(f"listen on:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=int(PORT))
