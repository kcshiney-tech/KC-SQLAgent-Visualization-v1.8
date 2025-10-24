#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试XLSX ROCE网络零件故障数据处理功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_xlsx_roce_network_part_fault():
    """测试XLSX ROCE网络零件故障数据处理功能"""
    try:
        # 导入相关模块
        from backend.dataSources.xlsx_roce_network_part_fault import process_xlsx_roce_network_part_fault
        from backend.data_loader import XLSXRoceNetworkPartFaultDataSourceLoader
        from backend.database_builder import DatabaseBuilder
        
        print("开始测试XLSX ROCE网络零件故障数据处理功能...")
        
        # 注意：这里需要一个真实的XLSX文件来进行测试
        # 在实际使用中，请将"test_data.xlsx"替换为真实的文件路径
        file_path = "D:\\Code\\KC-SQLAgent-Visualization-v1.8\\roce事件列表_2025-10-22 18_18_28.xlsx_1761128309.xlsx"
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"测试文件 {file_path} 不存在，请提供一个真实的XLSX文件进行测试")
            return
        
        # # 测试数据处理
        # print(f"\n1. 测试数据处理 ({file_path})...")
        # data = process_xlsx_roce_network_part_fault(file_path)
        # print(f"成功处理 {len(data)} 条记录")
        
        # if data:
        #     print("前3条记录:")
        #     for i, record in enumerate(data[:3]):
        #         print(f"  {i+1}. {record}")
        
        # # 测试数据加载器
        # print("\n2. 测试数据加载器...")
        # loader = XLSXRoceNetworkPartFaultDataSourceLoader(file_path)
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
        builder = DatabaseBuilder("test_xlsx_roce_network_part_fault.db")
        result = builder.build_database([XLSXRoceNetworkPartFaultDataSourceLoader(file_path)], rebuild=True)
        
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
    test_xlsx_roce_network_part_fault()
