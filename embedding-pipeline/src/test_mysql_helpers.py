from mysql_helpers import MYSQL_CLIENT, query_mysql_ops


def test_records_by_ids():
    ids = [442373625613648555, 442373625613648561, 442373625613648565]
    res = MYSQL_CLIENT.records_by_ids(ids)
    print(res)
    for it in res:
        print(it)


def test_query_mysql_ops():
    detail = query_mysql_ops(MYSQL_CLIENT)(442373625613648555)
    print(detail)
