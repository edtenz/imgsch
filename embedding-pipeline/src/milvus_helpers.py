import sys

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

from config import MILVUS_HOST, MILVUS_PORT, VECTOR_DIMENSION, METRIC_TYPE, INDEX_TYPE
from logger import LOGGER


class MilvusClient(object):

    def __init__(self, host: str = MILVUS_HOST, port: int = MILVUS_PORT):
        try:
            connections.connect(host=host, port=port)
            LOGGER.debug(f"Successfully connect to Milvus with IP:{host} and PORT:{port}")
        except Exception as e:
            LOGGER.error(f"Failed to connect Milvus: {e}")
            sys.exit(1)

    def has_collection(self, collection_name: str) -> bool:
        """
        Check collection exists or not
        :param collection_name: collection name in Milvus
        :return:
        """
        try:
            return utility.has_collection(collection_name)
        except Exception as e:
            LOGGER.error(f"Failed to load data to Milvus: {e}")
            return False

    def get_collection(self, collection_name: str) -> Collection:
        """
        Get collection by name
        :param collection_name: collection name in Milvus
        :return:
        """
        if self.has_collection(collection_name):
            return Collection(collection_name)
        raise Exception(f"Collection {collection_name} does not exist")

    def create_collection(self, collection_name: str, dim: int = VECTOR_DIMENSION) -> Collection:
        field1 = FieldSchema(name="id", dtype=DataType.INT64, descrition="int64", is_primary=True, auto_id=True)
        field2 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, descrition="float vector",
                             dim=dim, is_primary=False)
        schema = CollectionSchema(fields=[field1, field2], description="collection description")
        collection = Collection(name=collection_name, schema=schema)

        index_params = {
            'metric_type': METRIC_TYPE,
            'index_type': INDEX_TYPE,
            'params': {"nlist": dim}
        }
        status = collection.create_index(field_name='embedding', index_params=index_params)
        LOGGER.debug(
            f"Successfully create index in collection:{collection_name} with param:{index_params}, status: {status}")
        return collection

    def delete_collection(self, collection_name: str):
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            LOGGER.debug(f"Successfully drop collection: {collection_name}")
        return "ok"

    def insert(self, collection: Collection, vectors: list[float]) -> list[int]:
        # Batch insert vectors to milvus collection
        data = [vectors]
        mr = collection.insert(data)
        ids = mr.primary_keys
        collection.load()
        LOGGER.debug(
            f"Insert vectors to Milvus in collection: {collection_name} with {len(vectors)} rows")
        return ids

    def search_vectors(self, collection_name: str, vectors, top_k):
        # Search vector in milvus collection
        collection = self.get_collection(collection_name)
        search_params = {"metric_type": METRIC_TYPE, "params": {"nprobe": 192}}
        # data = [vectors]
        res = collection.search(vectors, anns_field="embedding", param=search_params, limit=top_k)
        print(res[0])
        LOGGER.debug(f"Successfully search in collection: {res}")
        return res

    def count(self, collection_name: str) -> int:
        # Get the number of milvus collection
        collection = self.get_collection(collection_name)
        num = collection.num_entities
        LOGGER.debug(f"Successfully get the num:{num} of the collection:{collection_name}")
        return num


MILVUS_CLIENT = MilvusClient()
