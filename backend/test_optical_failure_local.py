#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试光学故障数据加载和数据库构建
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from backend.data_loader import OpticalFailureDataSourceLoader
from backend.database_builder import DatabaseBuilder

def test_optical_failure_data_loading():
    """测试光学故障数据加载"""
    print("测试光学故障数据加载...")
    
    # 创建数据源加载器
    loader = OpticalFailureDataSourceLoader()
    
    try:
        # 加载数据
        data = loader.load_data()
        print(f"成功加载数据，共 {len(data)} 个表")
        
        if data:
            table_data = data[0]
            print(f"表名: {table_data['table_name']}")
            print(f"记录数: {len(table_data['data'])}")
            
            # 显示前几条记录
            print("\n前5条记录:")
            for i, record in enumerate(table_data['data'][:5]):
                print(f"  {i+1}. {record}")
        else:
            print("没有加载到数据")
            
    except Exception as e:
        print(f"数据加载失败: {e}")
        return False
        
    return True

def test_database_building():
    """测试数据库构建"""
    print("\n测试数据库构建...")
    
    # 创建数据源加载器列表
    data_sources = [OpticalFailureDataSourceLoader()]
    
    # 创建数据库构建器
    db_path = "custom_database.db"
    builder = DatabaseBuilder(db_path)
    
    try:
        # 构建数据库
        result = builder.build_database(data_sources, rebuild=True)
        
        if result["status"] == 0:
            print("数据库构建成功")
            if result["errors"]:
                print("警告信息:")
                for error in result["errors"]:
                    print(f"  - {error}")
        else:
            print("数据库构建失败")
            for error in result["errors"]:
                print(f"  - {error}")
            return False
            
    except Exception as e:
        print(f"数据库构建失败: {e}")
        return False
        
    return True

def main():
    """主函数"""
    print("开始测试光学故障数据处理...")
    
    # 测试数据加载
    if not test_optical_failure_data_loading():
        print("数据加载测试失败")
        return
    
    # 测试数据库构建
    if not test_database_building():
        print("数据库构建测试失败")
        return
    
    print("\n所有测试完成!")

if __name__ == "__main__":
    main()
