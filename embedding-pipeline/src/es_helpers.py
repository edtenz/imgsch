import sys

from elasticsearch import Elasticsearch, helpers

from config import (
    ES_HOST, ES_PORT, ES_INDEX,
)
from logger import LOGGER


class EsClient(object):

    def __init__(self, host: str = ES_HOST, port: int = ES_PORT):
        try:
            self.es = Elasticsearch(
                hosts=[f'http://{host}:{port}'],
            )
        except Exception as e:
            LOGGER.error(f"Failed to connect Elasticsearch: {e}")
            sys.exit(1)

    def exist_index(self, index_name: str) -> bool:
        """
        Check index exists or not
        :param index_name: index name in Elasticsearch
        :return:
        """
        return self.es.indices.exists(index=index_name)

    def create_index(self, index_name: str, body: dict) -> bool:
        """
        Create index with index name and doc
        :param index_name: index name in Elasticsearch
        :param body: index mappings
        :return:
        """
        if self.exist_index(index_name):
            LOGGER.debug(f"Index {index_name} already exists")
            return True
        try:
            self.es.indices.create(index=index_name, body=body)
            LOGGER.debug(f"Successfully create index: {index_name}")
            return True
        except Exception as e:
            LOGGER.error(f"Failed to create index: {e}")
            return False

    def insert_doc(self, index_name: str, doc: dict, id: str = None) -> bool:
        """
        Insert document
        :param index_name: index name in Elasticsearch
        :param doc: document doc
        :param id: document id
        :return:
        """
        try:
            self.es.index(index=index_name, document=doc, id=id)
            LOGGER.debug(f"Successfully insert document {id} to index: {index_name}")
            return True
        except Exception as e:
            LOGGER.error(f"Failed to insert document: {e}")
            return False

    def insert_batch(self, index_name: str, docs: list[dict]) -> int:
        """
        Insert batch documents
        :param index_name: index name in Elasticsearch
        :param docs: documents to insert, each document is a dict
            eg: {"_id": 1, "_source": {"name": "John Doe", "age": 30, "job": "Engineer"}},
        :return:
        """
        try:
            res = helpers.bulk(self.es, docs, index=index_name)
            return len(res)
        except Exception as e:
            LOGGER.error(f"Failed to insert documents: {e}")
            return 0

    def query(self, index_name: str, body: dict) -> list[dict]:
        """
        Query documents
        :param index_name: index name in Elasticsearch
        :param body: query body
        :return:
        """
        try:
            res = self.es.search(index=index_name, body=body)
            hits = res['hits']['hits']
            return hits
        except Exception as e:
            LOGGER.error(f"Failed to query documents: {e}")
            return []


def insert_doc_ops(es_cli: EsClient, index_name: str = ES_INDEX):
    def wrapper(doc: dict, id: str = None) -> bool:
        """
        Insert document to Elasticsearch
        :param doc: document to insert
        :param id: document id
        :return: true if insert successfully, false otherwise
        """
        return es_cli.insert_doc(index_name, doc, id)

    return wrapper


def insert_img_doc_ops(es_cli: EsClient, index_name: str = ES_INDEX):
    """
    Insert image document to Elasticsearch
    :param es_cli: es client
    :param index_name: index name
    :param (img_key, img_url, bbox, bbox_score, label, features, id)
    :return: true if insert successfully, false otherwise
    """

    def wrapper(img_key: str, img_url: str,
                bbox: str, bbox_score: float, label: str,
                features: list[float], id: str = None) -> bool:
        doc = {
            'image_key': img_key,
            'image_url': img_url,
            'bbox': bbox,
            'bbox_score': bbox_score,
            'label': label,
            'features': features,
        }
        return es_cli.insert_doc(index_name, doc, id)

    return wrapper


def bulk_docs_ops(es_cli: EsClient, index_name: str = ES_INDEX):
    def wrapper(docs: list[dict]) -> int:
        """
        Insert batch documents to Elasticsearch
        :param docs: documents to insert
        :return:
        """
        return es_cli.insert_batch(index_name, docs)

    return wrapper


def knn_query_docs_ops(es_cli: EsClient, index_name: str = ES_INDEX):
    def wrapper(vec: list[float], k: int = 10, num_candidates: int = 100) -> list[(str, str, str, float, str, float)]:
        """
        Query documents from Elasticsearch
        :param vec: query vector
        :param k: k nearest neighbors
        :param num_candidates: number of candidates
        :return: list of records: (image_key, image_url, bbox, bbox_score, label, score)
        """
        knn_query = {
            "knn": {
                "field": "features",
                "query_vector": vec,
                "k": k,
                "num_candidates": num_candidates
            },
            "_source": {
                "excludes": ["features"]
            }
        }

        hits = es_cli.query(index_name, knn_query)
        if len(hits) == 0:
            LOGGER.debug(f"No hits found for query: {knn_query}")
            return []
        res = []
        for hit in hits:
            source = hit['_source']
            res.append((
                source['image_key'],
                source['image_url'],
                source['bbox'],
                source['bbox_score'],
                source['label'],
                hit['_score'],
            ))

        return res

    return wrapper


def create_img_index(es_cli: EsClient, index_name: str = ES_INDEX) -> bool:
    body = {
        "settings": {
            "index": {
                "refresh_interval": "180s",
                "number_of_replicas": "0"
            }
        },
        "mappings": {
            "properties": {
                "image_key": {
                    "type": "keyword",
                    "index": True
                },
                "image_url": {
                    "type": "text",
                    "index": False
                },
                "bbox": {
                    "type": "keyword",
                    "index": False
                },
                "bbox_score": {
                    "type": "float",
                    "index": True
                },
                "label": {
                    "type": "text",
                    "index": True
                },
                "features": {
                    "type": "dense_vector",
                    "dims": 768,
                    "index": True,
                    "similarity": "dot_product",
                    "index_options": {
                        "type": "hnsw",
                        "m": 16,
                        "ef_construction": 256
                    }
                }
            }
        }
    }

    return es_cli.create_index(index_name=index_name, body=body)
