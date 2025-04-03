import pymysql

try:

    # conn = pymysql.connect(
    #             host='203.255.3.237',
    #             port='3306',
    #             user='root',
    #             password='mysql',
    #             db='junggal_v2_database',
    #             charset='utf8')

    print('here:)')
    conn = pymysql.connect(
                host='203.255.3.237',
                user='chaehyun',
                password='123456',
                db='junggal_v2_database',
                charset='utf8')

    # Connection 으로부터 Cursor 생성
    curs = conn.cursor()

    #SQL문 실행 및 Fetch
    sql = "update share_post set tired=1 where share_post_id=58;"
    # sql = "delete from share_post where share_post_id=1;"

    # user_id = 'admin@gmail.com'
    # user_id = 'lch0967'
    # feeling = [1, 0, 0, 1]
    # sql = "update share_post set good=1, sad=1, tired=1, stress=1 \
    #     where user_id = 'admin@gmail.com' order by post_time desc limit 1;"
    #         .format(feeling[0], feeling[1], feeling[2], feeling[3], user_id)

    # sql = "select * from share_post;"

    curs.execute(sql)
    print('success')
    # 데이터 fetch
    rows = curs.fetchall()
    for row in rows:
        print(row)

    conn.close()
    print('db connection success')


except:
    print("error!")