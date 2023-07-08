from config import ES_HOST, ES_PORT, ES_INDEX, MINIO_BUCKET_NAME
from es_helpers import EsClient, create_img_index
from es_load import embedding_es_pipe, do_es_embedding
from model import VitBase224


def test_embedding_pipeline():
    es_cli = EsClient(host=ES_HOST, port=ES_PORT)
    index_name = ES_INDEX
    ok = create_img_index(es_cli, index_name)
    print("ok: ", ok)
    if not ok:
        return

    img_url = 'http://localhost:10086/file/imgsch/0ad09b3fdb4fbd661923fb5555c7b341.jpg'
    model = VitBase224()
    res = embedding_es_pipe(img_url, model, es_cli, index_name)
    print("res: ", res)


def test_do_embedding():
    es_cli = EsClient(host=ES_HOST, port=ES_PORT)
    index_name = ES_INDEX
    model = VitBase224()
    test_count = 10000
    suc = do_es_embedding(MINIO_BUCKET_NAME, model, es_cli, index_name, test_count)
    print("suc: ", suc)
