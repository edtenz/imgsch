import requests
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

import image_helper
from config import (
    HTTP_PORT,
    MINIO_PROXY_ENDPOINT,
)
from load import do_milvus_embedding
from logger import LOGGER
from milvus_helpers import MilvusClient
from model import VitBase224
from mysql_helpers import MysqlClient
from search import do_search

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
MYSQL_CLIENT = MysqlClient()
MILVUS_CLIENT = MilvusClient()


@app.get("/ping")
def ping():
    LOGGER.debug("ping")
    return {"message": "pong"}


@app.get("/test")
def ping(q: str):
    LOGGER.debug(f"test, keywords: {q}")
    return [
        {"id": 1, "name": "test1", "url": "http://localhost:10086/file/imgsch/008e7a0c7582d987b183a59133f7169e.jpg"},
        {"id": 2, "name": "test2", "url": "http://localhost:10086/file/imgsch/008e7a0c7582d987b183a59133f7169e.jpg"},
    ]


@app.get('/load')
def load_img(img_bucket: str, table_name: str):
    try:
        LOGGER.debug(f"detect image bucket: {img_bucket}, table_name: {table_name}")
        count = do_milvus_embedding(img_bucket, MODEL, MILVUS_CLIENT, MYSQL_CLIENT, table_name)
        return JSONResponse({'status': True, 'msg': 'success', 'data': count})
    except Exception as e:
        LOGGER.error(f"Get image error: {e}")
        return {'status': False, 'msg': e}, 400


@app.post("/search")
async def search(file: UploadFile = File(...)):
    contents = await file.read()
    resize_img = image_helper.thumbnail_bytes(contents, 450, 60)
    key = image_helper.md5_content(resize_img)

    files = {
        'file': (key, resize_img, 'image/jpeg'),
    }

    bucket_name = 'search'
    upload_url = f'http://{MINIO_PROXY_ENDPOINT}/file/{bucket_name}/{key}'
    response = requests.post(upload_url, files=files)
    if response.status_code != 200:
        return JSONResponse({'status': False, 'msg': 'upload image failed'})

    img_url = upload_url
    obj_feat, candidate_box, res_list = do_search(img_url, MODEL, MILVUS_CLIENT, MYSQL_CLIENT)
    if obj_feat is None:
        return JSONResponse({'status': False, 'msg': 'image detect or extract failed'})
    if len(res_list) == 0:
        return JSONResponse({'status': False, 'msg': 'no result found'})
    res_list = res_list[:10]

    data = {
        'search_img': obj_feat.url,
        'bbox': obj_feat.bbox.to_dict(),
        'candidate_box': [item.to_dict() for item in candidate_box],
        'results': [item.to_dict() for item in res_list]
    }

    return JSONResponse({'status': True, 'msg': 'success', 'data': data})


def start_http_server():
    LOGGER.info(f"http listen on: {HTTP_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=int(HTTP_PORT))
