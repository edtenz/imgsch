import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from config import DEFAULT_TABLE
from config import HTTP_PORT
from load import do_load
from logger import LOGGER
from milvus_helpers import MilvusClient
from minio_helpers import MinioClient
from model import Vit224
from mysql_helpers import MysqlClient

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
MODEL = Vit224()
MILVUS_CLI = MilvusClient()
MYSQL_CLI = MysqlClient()
MINIO_CLI = MinioClient()


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
