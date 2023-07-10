from pydantic import BaseModel
from towhee import pipe

from config import (
    DEFAULT_TABLE,
    ES_INDEX,
)
from es_helpers import EsClient, knn_query_docs_ops
from logger import LOGGER
from milvus_helpers import MilvusClient, search_milvus_ops
from model import ImageFeatureModel, ObjectFeature, BoundingBox
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


def do_milvus_search(img_url: str,
                     model: ImageFeatureModel,
                     milvus_client: MilvusClient,
                     mysql_cli: MysqlClient,
                     table_name: str = DEFAULT_TABLE) -> (ObjectFeature, list[BoundingBox], list[SearchResult]):
    """
    Search similar images for the given image.
    :param img_url: given image path
    :param model: model instance
    :param milvus_client: milvus client
    :param mysql_cli: mysql client
    :param table_name: table name
    :return: list of similar images: [(image_url, similarity), ...]
    """
    obj_feat, candidate_box = model.extract_primary_features(img_url)
    if obj_feat is None:
        return None, candidate_box, []
    if obj_feat.features is None:
        return obj_feat, candidate_box, []

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
    return obj_feat, candidate_box, res_list


def do_es_search(img_url: str,
                 model: ImageFeatureModel,
                 es_cli: EsClient,
                 index_name: str = ES_INDEX) -> (ObjectFeature, list[BoundingBox], list[SearchResult]):
    obj_feat, candidate_box = model.extract_primary_features(img_url)
    if obj_feat is None:
        return None, candidate_box, []
    if obj_feat.features is None:
        return obj_feat, candidate_box, []

    p_search_pre = (
        pipe.input('vec').
        filter(('vec',), ('vec',), 'vec', lambda x: x is not None and len(x) > 0).
        map('vec', 'k', lambda x: 10).
        map('vec', 'candidates', lambda x: 20).
        flat_map(('vec', 'k', 'candidates'),
                 ('image_key', 'image_url', 'bbox', 'bbox_score', 'label', 'score'),
                 knn_query_docs_ops(es_cli, index_name)).
        filter(('image_url', 'bbox', 'label', 'score'),
               ('image_url', 'bbox', 'label', 'score'),
               'score', lambda x: x > 0.65).
        output('image_url', 'bbox', 'label', 'score')
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
    return obj_feat, candidate_box, res_list
