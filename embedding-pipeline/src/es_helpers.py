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
            return res['hits']['hits']
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
    def wrapper(vec: list[float], k: int = 10, num_candidates: int = 100) -> list[dict]:
        """
        Query documents from Elasticsearch
        :param vec: query vector
        :param k: k nearest neighbors
        :param num_candidates: number of candidates
        :return:
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

        res = es_cli.query(index_name, knn_query)
        return res

    return wrapper
