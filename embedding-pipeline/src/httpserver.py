import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from config import DEFAULT_TABLE, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_ADDR
from config import HTTP_PORT
from load import do_load
from logs import LOGGER
from milvus_helpers import MilvusHelper
from minio_helpers import MinioHelper
from model import Resnet50
from mysql_helpers import MySQLHelper

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)
MODEL = Resnet50()
MILVUS_CLI = MilvusHelper()
MYSQL_CLI = MySQLHelper()
MINIO_CLI = MinioHelper(MINIO_ADDR, MINIO_ACCESS_KEY, MINIO_SECRET_KEY)


@app.get("/ping")
def ping():
    LOGGER.debug("ping")
    return {"message": "pong"}


@app.get('/load')
def load_img(img_dir: str):
    try:
        LOGGER.debug(f"detect image: {img_dir}")
        count = do_load(DEFAULT_TABLE, img_dir, MODEL, MILVUS_CLI, MYSQL_CLI, MINIO_CLI)
        return JSONResponse({'status': True, 'msg': 'success', 'data': count})
    except Exception as e:
        LOGGER.error(f"Get image error: {e}")
        return {'status': False, 'msg': e}, 400


def start_http_server():
    LOGGER.info(f"http listen on: {HTTP_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=int(HTTP_PORT))
