import shutil

import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

from config import HTTP_PORT, MINIO_DOWNLOAD_PATH
from image_helper import thumbnail
from load import do_load
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
def load_img(img_dir: str, table_name: str):
    try:
        LOGGER.debug(f"detect image: {img_dir}, table_name: {table_name}")
        count = do_load(img_dir, MODEL, MILVUS_CLIENT, MYSQL_CLIENT, MINIO_CLIENT, table_name)
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
    img_path = f"{MINIO_DOWNLOAD_PATH}/{file.filename}"
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    resized_img = thumbnail(img_path, 450, f'{MINIO_DOWNLOAD_PATH}/thumbnail/{file.filename}', 60)
    obj_feat, res_list = do_search(resized_img, MODEL, MILVUS_CLIENT, MYSQL_CLIENT)
    if obj_feat is None:
        return JSONResponse({'status': False, 'msg': 'image detect or extract failed'})
    if len(res_list) == 0:
        return JSONResponse({'status': False, 'msg': 'no result found'})
    res_list = res_list[:10]
    upload_res = MINIO_CLIENT.upload(obj_feat.key, resized_img)

    data = {
        'search_img': obj_feat.key,
        'bbox': list(obj_feat.box),
        'label': obj_feat.label,
        'score': obj_feat.score,
        'results': [item.to_dict() for item in res_list]
    }

    return JSONResponse({'status': True, 'msg': 'success', 'data': data})


def start_http_server():
    LOGGER.info(f"http listen on: {HTTP_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=int(HTTP_PORT))
