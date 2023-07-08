from mysql_helpers import MysqlClient, query_mysql_ops

MYSQL_CLIENT = MysqlClient()


def test_records_by_ids():
    ids = [442373625613648555, 442373625613648561, 442373625613648565]
    res = MYSQL_CLIENT.records_by_ids(ids)
    print(res)
    for it in res:
        print(it)


def test_query_mysql_ops():
    detail = query_mysql_ops(MYSQL_CLIENT)(442373625613648555)
    print(detail)


def test_scan_table():
    res = MYSQL_CLIENT.scan_table()
    print("res:", len(res))
    for it in res:
        # print(it)
        vec_id = it[0]
        img_key = it[1]
        box = it[2]
        score = it[3]
        label = it[4]
        # print(vec_id, img_key, box, score, label)
        # replace img_key string "api" into "file"
        img_key = img_key.replace("api", "file")
        print(vec_id, img_key, box, score, label)
        MYSQL_CLIENT.update_by_id(vec_id, img_key, box, score, label)
