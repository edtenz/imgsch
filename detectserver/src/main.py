import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from config import PORT
from detect import detect
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
        bboxes = []
        for bbox in detect(img):
            bbox_map = {
                'box': bbox.box,
                'score': bbox.score,
                'cat': bbox.cat
            }
            bboxes.append(bbox_map)
        LOGGER.info(f"Successfully detect image: {img}, bboxes: {bboxes}")
        return JSONResponse({'status': True, 'msg': 'success', 'data': bboxes})
    except Exception as e:
        LOGGER.error(f"Get image error: {e}")
        return {'status': False, 'msg': e}, 400


if __name__ == '__main__':
    print('Start image detect service...')
    LOGGER.info(f"listen on:{PORT}")
    # LOGGER.debug('debug message')
    # LOGGER.error('error message')

    uvicorn.run(app, host="0.0.0.0", port=int(PORT))
