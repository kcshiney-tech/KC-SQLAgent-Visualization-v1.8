#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试光模块库存数据功能
"""

import sys
import os
import traceback

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from backend.dataSources.optical_module_inventory import query_optical_module_inventory
from backend.data_loader import OpticalModuleInventoryDataSourceLoader
from backend.database_builder import DatabaseBuilder

def test_optical_module_inventory():
    """测试光模块库存数据获取功能"""
    print("测试光模块库存数据获取功能...")
    
    try:
        # 获取数据
        data = query_optical_module_inventory()
        print(f"成功获取数据，共 {len(data)} 条记录")
        
        if data:
            print("\n前5条记录:")
            for i, record in enumerate(data[:5]):
                print(f"  {i+1}. {record}")
        else:
            print("没有获取到数据")
            
    except Exception as e:
        print(f"测试失败: {e}")
        traceback.print_exc()

def test_data_loader():
    """测试数据加载器"""
    print("\n测试数据加载器...")
    
    try:
        # 创建数据加载器
        loader = OpticalModuleInventoryDataSourceLoader()
        
        # 加载数据
        data = loader.load_data()
        print(f"数据加载器返回 {len(data)} 个表的数据")
        
        if data:
            for table_info in data:
                table_name = table_info["table_name"]
                table_data = table_info["data"]
                print(f"  表名: {table_name}")
                print(f"  记录数: {len(table_data)}")
                if table_data:
                    print(f"  第一条记录: {table_data[0]}")
        else:
            print("数据加载器没有返回数据")
            
    except Exception as e:
        print(f"数据加载器测试失败: {e}")
        traceback.print_exc()

def test_database_builder():
    """测试数据库构建器"""
    print("\n测试数据库构建器...")
    
    try:
        # 创建数据加载器
        loader = OpticalModuleInventoryDataSourceLoader()
        
        # 创建数据库构建器
        builder = DatabaseBuilder("test_optical_module_inventory.db")
        
        # 构建数据库
        result = builder.build_database([loader], rebuild=True)
        print(f"数据库构建结果: status={result['status']}, errors={result['errors']}")
        
        if result['status'] == 0:
            print("数据库构建成功")
        else:
            print("数据库构建失败")
            
    except Exception as e:
        print(f"数据库构建器测试失败: {e}")
        traceback.print_exc()

def main():
    """主函数"""
    print("开始测试光模块库存数据功能...")
    
    test_optical_module_inventory()
    test_data_loader()
    test_database_builder()
    
    print("\n所有测试完成!")

if __name__ == "__main__":
    main()
