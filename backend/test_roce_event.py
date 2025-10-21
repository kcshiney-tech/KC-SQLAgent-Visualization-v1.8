#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试ROCE事件数据源
"""

import sys
import os
import sqlite3

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from backend.data_loader import RoceEventDataSourceLoader
from backend.database_builder import DatabaseBuilder

def test_roce_event_data_source():
    """测试ROCE事件数据源"""
    print("测试ROCE事件数据源...")
    
    try:
        # 创建数据加载器
        loader = RoceEventDataSourceLoader()
        
        # 加载数据
        data = loader.load_data()
        print(f"数据加载完成，共加载 {len(data)} 个表")
        
        if data:
            table_data = data[0]
            print(f"表名: {table_data['table_name']}")
            print(f"记录数: {len(table_data['data'])}")
            print("前5条记录:")
            for i, record in enumerate(table_data['data'][:5]):
                print(f"  {i+1}. {record}")
        else:
            print("未能加载数据")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_roce_event_database():
    """测试ROCE事件数据库"""
    print("\n测试ROCE事件数据库...")
    
    try:
        # 创建数据加载器
        loader = RoceEventDataSourceLoader()
        
        # 创建数据库构建器
        builder = DatabaseBuilder("test_roce_event.db")
        
        # 构建数据库
        result = builder.build_database([loader], rebuild=True)
        print(f"数据库构建结果: status={result['status']}, errors={result['errors']}")
        
        if result['status'] == 0:
            print("数据库构建成功")
            
            # 检查数据库中的表
            conn = sqlite3.connect("test_roce_event.db")
            cursor = conn.cursor()
            
            # 查询所有表名
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print("数据库中的表:")
            for table in tables:
                print(f"  {table[0]}")
                
            # 查询ROCE网络事件-光模块故障表的前几条记录
            cursor.execute("SELECT * FROM 'ROCE网络事件-网络零件（光模块+AOC）故障表' LIMIT 5;")
            records = cursor.fetchall()
            print("\nROCE网络事件-网络零件（光模块+AOC）故障表的前5条记录:")
            # 获取列名
            cursor.execute("PRAGMA table_info('ROCE网络事件-网络零件（光模块+AOC）故障表');")
            columns = [column[1] for column in cursor.fetchall()]
            print("列名:", columns)
            for record in records:
                print("  ", record)
                
            conn.close()
        else:
            print("数据库构建失败")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("开始测试ROCE事件数据源...")
    
    test_roce_event_data_source()
    test_roce_event_database()
    
    print("\n测试完成!")

if __name__ == "__main__":
    main()
