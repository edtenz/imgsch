import sys

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, SearchResult

from config import MILVUS_HOST, MILVUS_PORT, VECTOR_DIMENSION, METRIC_TYPE, INDEX_TYPE, DEFAULT_TABLE, NLIST
from logger import LOGGER


class MilvusClient(object):

    def __init__(self, host: str = MILVUS_HOST, port: int = MILVUS_PORT):
        try:
            connections.connect(host=host, port=port)
            LOGGER.debug(f"Successfully connect to Milvus with IP:{host} and PORT:{port}")
        except Exception as e:
            LOGGER.error(f"Failed to connect Milvus: {e}")
            sys.exit(1)

    def exist_collection(self, collection_name: str = DEFAULT_TABLE) -> bool:
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

    def create_collection(self, collection_name: str = DEFAULT_TABLE,
                          dim: int = VECTOR_DIMENSION) -> Collection:
        if self.delete_collection(collection_name):
            LOGGER.debug(f"Successfully drop collection: {collection_name}")

        fields = [
            FieldSchema(name='id', dtype=DataType.INT64, description='image embedding id', is_primary=True,
                        auto_id=True),
            FieldSchema(name='vec', dtype=DataType.FLOAT_VECTOR, description='image embedding vectors', dim=dim,
                        is_primary=False)
        ]
        schema = CollectionSchema(fields=fields, description='reverse image search')
        collection = Collection(name=collection_name, schema=schema)

        if self.create_index(collection):
            LOGGER.debug(f"Successfully create collection: {collection_name}")
        return collection

    def create_index(self, collection: Collection, index_params: dict = None) -> bool:
        """
        Create index for collection
        :param collection: collection object
        :param index_params: index parameters
        :return: True if create index successfully, False otherwise
        """
        if index_params is None:
            index_params = {
                'metric_type': METRIC_TYPE,
                'index_type': INDEX_TYPE,
                'params': {"nlist": NLIST}
            }
        collection.create_index(field_name='vec', index_params=index_params)
        return True

    def insert_batch(self, collection_name: str, vectors: list[list[float]]) -> list[int]:
        """
        Insert vectors to Milvus
        :param collection_name: collection name in Milvus
        :param vectors: vectors to insert
        :return: primary keys of vectors
        """
        try:
            collection = self.get_collection(collection_name)
            data = [vectors]
            mr = collection.insert(data)
            ids = mr.primary_keys
            collection.load()
            LOGGER.debug(
                f"Insert vectors to Milvus in collection: {collection_name} with {len(vectors)} rows")
            return ids
        except Exception as e:
            raise Exception(f"Failed to load data to Milvus: {e}")

    def insert(self, collection_name: str, vector: list[float]) -> int:
        """
        Insert vector to Milvus
        :param collection_name: collection name in Milvus
        :param vector: vector to insert
        :return: primary keys of vector
        """
        try:
            vectors = [vector]
            ids = self.insert_batch(collection_name, vectors)
            if len(ids) == 1:
                return ids[0]
            LOGGER.error(f"Failed to insert vector to Milvus: {vector}")
            return -1
        except Exception as e:
            LOGGER.error(f"Failed to load data to Milvus: {e}")
            return -1

    def get_collection(self, collection_name: str) -> Collection:
        """
        Get collection by name
        :param collection_name: collection name in Milvus
        :return:
        """
        if self.exist_collection(collection_name):
            return Collection(collection_name)
        raise Exception(f"Collection {collection_name} does not exist")

    def delete_collection(self, collection_name: str) -> bool:
        if self.exist_collection(collection_name):
            utility.drop_collection(collection_name)
            LOGGER.debug(f"Successfully drop collection: {collection_name}")
        return True

    def count(self, collection_name: str = DEFAULT_TABLE) -> int:
        # Get the number of milvus collection
        collection = self.get_collection(collection_name)
        num = collection.num_entities
        LOGGER.debug(f"Successfully get the num:{num} of the collection:{collection_name}")
        return num

    def search_vectors(self, collection_name: str = DEFAULT_TABLE, vectors: list[float] = None,
                       top_k: int = 10) -> SearchResult:
        # Search vector in milvus collection
        try:
            collection = self.get_collection(collection_name)
            search_params = {"metric_type": METRIC_TYPE, "params": {"nprobe": 16}}
            data = [vectors]
            res = collection.search(data, anns_field="vec", param=search_params, limit=top_k)
            LOGGER.debug(f"Successfully search in collection: {res}")
            return res
        except Exception as e:
            LOGGER.error(f"Failed to search vectors in Milvus: {e}")
            return None


def insert_milvus_ops(milvus_cli: MilvusClient, collection_name: str = DEFAULT_TABLE) -> callable:
    def wrapper(vector: list[float]) -> int:
        return milvus_cli.insert(collection_name, vector)

    return wrapper


MILVUS_CLIENT = MilvusClient()
