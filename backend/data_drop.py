import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.database_builder import DatabaseBuilder

# 创建DatabaseBuilder实例
builder = DatabaseBuilder("custom_database.db")

# 删除指定的数据表
table_names = ["ROCE网络事件-网络零件（光模块+AOC）故障表"]
result = builder.drop_tables(table_names)

# 检查结果
if result['status'] == 0:
    print("成功删除数据表")
else:
    print(f"删除数据表时出现错误: {result['errors']}")
