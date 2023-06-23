import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from config import HTTP_PORT
from load import do_load
from logger import LOGGER
from milvus_helpers import MILVUS_CLIENT
from minio_helpers import MINIO_CLIENT
from model import VitBase224
from mysql_helpers import MYSQL_CLIENT

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

# MODEL = Resnet50()
MODEL = VitBase224()


@app.get("/ping")
def ping():
    LOGGER.debug("ping")
    return {"message": "pong"}


@app.get('/load')
def load_img(img_dir: str, table_name: str):
    try:
        LOGGER.debug(f"detect image: {img_dir}, table_name: {table_name}")
        count = do_load(img_dir, MODEL, MILVUS_CLIENT, MYSQL_CLIENT, MINIO_CLIENT, table_name)
        return JSONResponse({'status': True, 'msg': 'success', 'data': count})
    except Exception as e:
        LOGGER.error(f"Get image error: {e}")
        return {'status': False, 'msg': e}, 400




@app.get('/img')
def get_img(image_key: str):
    try:
        LOGGER.debug(f"detect image: {image_key}")
        if not MINIO_CLIENT.exists_object(image_key):
            return {'status': False, 'msg': 'image not found'}, 404
        image_path = MINIO_CLIENT.download(image_key)
        return JSONResponse({'status': True, 'msg': 'success', 'data': ""})
    except Exception as e:
        LOGGER.error(f"Get image error: {e}")
        return {'status': False, 'msg': e}, 400


def start_http_server():
    LOGGER.info(f"http listen on: {HTTP_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=int(HTTP_PORT))
