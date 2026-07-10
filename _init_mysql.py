import pymysql

conn = pymysql.connect(host='localhost', port=3306, user='root', password='root', charset='utf8mb4')
cur = conn.cursor()

# 如果 airline_analytics 存在就删掉重建
cur.execute('DROP DATABASE IF EXISTS airline_analytics')
cur.execute('CREATE DATABASE airline_analytics DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci')
conn.commit()

cur.execute("SHOW DATABASES LIKE 'airline_analytics'")
row = cur.fetchone()
print('airline_analytics 已重建:', row is not None)

cur.close()
conn.close()
