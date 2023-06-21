import sys

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

from config import MILVUS_HOST, MILVUS_PORT, VECTOR_DIMENSION, METRIC_TYPE, INDEX_TYPE, DEFAULT_TABLE
from logger import LOGGER


class MilvusClient(object):

    def __init__(self, host: str = MILVUS_HOST, port: int = MILVUS_PORT):
        try:
            self.collection = None
            connections.connect(host=host, port=port)
            LOGGER.debug(f"Successfully connect to Milvus with IP:{host} and PORT:{port}")
        except Exception as e:
            LOGGER.error(f"Failed to connect Milvus: {e}")
            sys.exit(1)

    def has_collection(self, collection_name: str = DEFAULT_TABLE) -> bool:
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
            FieldSchema(name='key', dtype=DataType.VARCHAR, description='image key', max_length=50,
                        is_primary=True, auto_id=False),
            FieldSchema(name='vec', dtype=DataType.FLOAT_VECTOR, description='image embedding vectors', dim=dim)
        ]
        schema = CollectionSchema(fields=fields, description='reverse image search')
        collection = Collection(name=collection_name, schema=schema)

        index_params = {
            'metric_type': METRIC_TYPE,
            'index_type': INDEX_TYPE,
            'params': {"nlist": dim}
        }
        collection.create_index(field_name='vec', index_params=index_params)
        self.collection = collection
        return collection

    def get_collection(self, collection_name: str) -> Collection:
        """
        Get collection by name
        :param collection_name: collection name in Milvus
        :return:
        """
        if self.has_collection(collection_name) and self.collection is not None:
            return self.collection
        raise Exception(f"Collection {collection_name} does not exist")

    def delete_collection(self, collection_name: str) -> bool:
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            LOGGER.debug(f"Successfully drop collection: {collection_name}")
        return True

    def count(self, collection_name: str = DEFAULT_TABLE) -> int:
        # Get the number of milvus collection
        collection = self.get_collection(collection_name)
        num = collection.num_entities
        LOGGER.debug(f"Successfully get the num:{num} of the collection:{collection_name}")
        return num


MILVUS_CLIENT = MilvusClient()
