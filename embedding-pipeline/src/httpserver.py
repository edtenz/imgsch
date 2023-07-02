import requests
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

import image_helper
from config import HTTP_PORT, MINIO_PROXY_ENDPOINT
from load import do_embedding
from logger import LOGGER
from milvus_helpers import MILVUS_CLIENT
from minio_helpers import MINIO_CLIENT, download_object
from model import VitBase224
from mysql_helpers import MYSQL_CLIENT
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


@app.get("/ping")
def ping():
    LOGGER.debug("ping")
    return {"message": "pong"}


@app.get('/load')
def load_img(img_bucket: str, table_name: str):
    try:
        LOGGER.debug(f"detect image bucket: {img_bucket}, table_name: {table_name}")
        count = do_embedding(img_bucket, MODEL, MILVUS_CLIENT, MYSQL_CLIENT, table_name)
        return JSONResponse({'status': True, 'msg': 'success', 'data': count})
    except Exception as e:
        LOGGER.error(f"Get image error: {e}")
        return {'status': False, 'msg': e}, 400


@app.get('/img/{image_key}')
def get_img(image_key: str):
    try:
        LOGGER.debug(f"detect image: {image_key}")
        if not MINIO_CLIENT.exists_object(image_key):
            return {'status': False, 'msg': 'image not found'}, 404
        image_path = download_object(MINIO_CLIENT, image_key)
        if image_path is None or image_path == "":
            return {'status': False, 'msg': 'image not found'}, 404

        return FileResponse(image_path, media_type="image/jpeg")
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
    obj_feat, res_list = do_search(img_url, MODEL, MILVUS_CLIENT, MYSQL_CLIENT)
    if obj_feat is None:
        return JSONResponse({'status': False, 'msg': 'image detect or extract failed'})
    if len(res_list) == 0:
        return JSONResponse({'status': False, 'msg': 'no result found'})
    res_list = res_list[:10]

    data = {
        'search_img': obj_feat.url,
        'bbox': list(obj_feat.box),
        'label': obj_feat.label,
        'bbox_score': obj_feat.score,
        'results': [item.to_dict() for item in res_list]
    }

    return JSONResponse({'status': True, 'msg': 'success', 'data': data})


def start_http_server():
    LOGGER.info(f"http listen on: {HTTP_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=int(HTTP_PORT))
