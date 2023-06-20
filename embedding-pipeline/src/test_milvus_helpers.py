from milvus_helpers import MILVUS_CLIENT
from model import Vit224


def test_search_vectors():
    vit_model = Vit224()
    vectors = vit_model.extract_features('../data/test.jpg')
    print('vector size:', len(vectors))
    res = MILVUS_CLIENT.search_vectors('milvus_imgsch_tab', vectors, 10)
    print(res)
