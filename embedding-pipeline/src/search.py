from pydantic import BaseModel
from towhee import pipe

from config import DEFAULT_TABLE, MINIO_BUCKET_NAME
from logger import LOGGER
from milvus_helpers import MilvusClient, search_milvus_ops
from minio_helpers import MinioClient, download_minio_ops
from model import Model, ObjectFeature
from mysql_helpers import MysqlClient, query_mysql_ops


class SearchResult(BaseModel):
    image_key: str
    box: str
    label: str
    score: float

    def __str__(self):
        return f"image_key: {self.image_key}, box: {self.box}, label: {self.label}, score: {self.score}"

    def to_dict(self):
        return {
            "image_key": self.image_key,
            "box": self.box,
            "label": self.label,
            "score": self.score
        }


def do_search(img_url: str,
              model: Model,
              milvus_client: MilvusClient,
              mysql_cli: MysqlClient,
              table_name: str = DEFAULT_TABLE) -> (ObjectFeature, list[SearchResult]):
    """
    Search similar images for the given image.
    :param img_url: given image path
    :param model: model instance
    :param milvus_client: milvus client
    :param mysql_cli: mysql client
    :param table_name: table name
    :return: list of similar images: [(image_url, similarity), ...]
    """
    obj_feat = model.extract_primary_features(img_url)
    if obj_feat is None:
        return None, []

    p_search_pre = (
        pipe.input('vec')
        .flat_map('vec', ('vec_id', 'distance'), search_milvus_ops(milvus_client, table_name, 10))
        .map('vec_id', ('id', 'image_key', 'box', 'score', 'label'), query_mysql_ops(mysql_cli, table_name))
        .output('image_key', 'box', 'label', 'distance')
    )

    res = p_search_pre(obj_feat.features)
    size = res.size
    LOGGER.info(f"Search result size: {size}")
    if size == 0:
        return obj_feat, []
    res_list = []
    for i in range(size):
        it = res.get()
        search_res = SearchResult(image_key=it[0], box=it[1], label=it[2], score=it[3])
        res_list.append(search_res)
    return obj_feat, res_list


def do_download(image_key: str,
                minio_client: MinioClient) -> str:
    """
    Download image from minio by give image key.
    :param image_key:
    :param minio_client:
    :param table_name:
    :return:
    """

    p_download = (
        pipe.input('image_key')
        .map('image_key', 'img_path', download_minio_ops(minio_client, MINIO_BUCKET_NAME))
        .filter(('img_path'), ('img_path'), 'img_path', lambda url: (url is not None or url != ''))
        .output('img_path')
    )

    res = p_download(image_key)
    size = res.size
    LOGGER.info(f"Download result size: {size}")
    if size == 0:
        raise Exception(f"Download image failed: {image_key}")
    return res.get()[0]
