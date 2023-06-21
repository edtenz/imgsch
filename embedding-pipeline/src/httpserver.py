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
def load_img(img_dir: str):
    try:
        LOGGER.debug(f"detect image: {img_dir}")
        count = do_load(img_dir, MODEL, MILVUS_CLIENT, MYSQL_CLIENT, MINIO_CLIENT)
        return JSONResponse({'status': True, 'msg': 'success', 'data': count})
    except Exception as e:
        LOGGER.error(f"Get image error: {e}")
        return {'status': False, 'msg': e}, 400


def start_http_server():
    LOGGER.info(f"http listen on: {HTTP_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=int(HTTP_PORT))
