import os

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from config import HTTP_PORT, MINIO_DOWNLOAD_PATH
from detect import Detector
from logs import LOGGER
from minio_client import md5_hash, mclient, remove_local_object

app = FastAPI()
detector = Detector()


@app.get("/ping")
def ping():
    LOGGER.debug("ping")
    return {"message": "pong"}


@app.get('/detect')
def detect_img(key: str):
    try:
        LOGGER.debug(f"detect image: {key}")
        bboxes = [item.to_dict() for item in detector.detect(key)]
        LOGGER.info(f"Successfully detect image: {key}, bboxes: {bboxes}")
        return JSONResponse({'status': True, 'msg': 'success', 'data': bboxes})
    except Exception as e:
        LOGGER.error(f"Get image error: {e}")
        return {'status': False, 'msg': e}, 400


@app.post('/img/upload')
async def upload_images(image: UploadFile = File(None)):
    try:
        if image is not None:
            content = await image.read()
            LOGGER.info(f"received image: {image.filename}")
            object_name = md5_hash(content)
            img_path = os.path.join(MINIO_DOWNLOAD_PATH, image.filename)
            with open(img_path, "wb+") as f:
                f.write(content)
        else:
            return {'status': False, 'msg': 'Image and url are required'}, 400
        mclient.upload(object_name, img_path)
        LOGGER.info(f"Successfully uploaded data, object name: {object_name}")
        remove_local_object(img_path)
        return {'status': True, 'msg': 'success', 'data': object_name}
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


def start_http_server():
    LOGGER.info(f"listen on:{HTTP_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=int(HTTP_PORT))
