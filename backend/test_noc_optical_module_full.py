#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试NOC光模块全量数据拉取功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_noc_optical_module_full():
    """测试NOC光模块全量数据拉取功能"""
    try:
        # 导入相关模块
        from backend.dataSources.api_optical_module_full_query import fetch_and_process_optical_modules_full
        from backend.data_loader import NOCOpticalModuleFullDataSourceLoader
        from backend.database_builder import DatabaseBuilder
        
        print("开始测试NOC光模块全量数据拉取功能...")
        
        # # 测试数据拉取和处理
        # print("\n1. 测试数据拉取和处理...")
        # data = fetch_and_process_optical_modules_full()
        # print(f"成功拉取并处理 {len(data)} 条记录")
        
        # if data:
        #     print("前3条记录:")
        #     for i, record in enumerate(data[:3]):
        #         print(f"  {i+1}. {record}")
        
        # # 测试数据加载器
        # print("\n2. 测试数据加载器...")
        # loader = NOCOpticalModuleFullDataSourceLoader()
        # loaded_data = loader.load_data()
        # print(f"数据加载器返回 {len(loaded_data)} 个表")
        
        # if loaded_data:
        #     table_data = loaded_data[0]
        #     print(f"表名: {table_data['table_name']}")
        #     print(f"记录数: {len(table_data['data'])}")
        #     if table_data['data']:
        #         print("前3条记录:")
        #         for i, record in enumerate(table_data['data'][:3]):
        #             print(f"  {i+1}. {record}")
        
        # 测试数据库构建
        print("\n3. 测试数据库构建...")
        builder = DatabaseBuilder("test_noc_optical_module.db")
        result = builder.build_database([NOCOpticalModuleFullDataSourceLoader()], rebuild=True)
        
        if result['status'] == 0:
            print("数据库构建成功")
        else:
            print(f"数据库构建失败: {result['errors']}")
            
        print("\n测试完成!")
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_noc_optical_module_full()
