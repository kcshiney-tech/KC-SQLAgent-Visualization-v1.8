#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试光模块API查询工具
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.dataSources.api_optical_module_query import query_optical_modules, save_to_csv

def test_api_query():
    """测试API查询功能"""
    print("测试光模块API查询功能...")
    
    # 测试数据
    batch_query_content = """CVJH05023511277 
210231A562N144011744"""
    
    # 执行查询
    result = query_optical_modules(batch_query_content)
    
    # 检查结果
    if "error" in result:
        print(f"查询失败: {result['error']}")
        return False
    else:
        data = result.get("data", [])
        print(f"查询成功，共获得 {len(data)} 条记录")
        
        # 显示前几条记录
        if data:
            print("前2条记录:")
            for i, record in enumerate(data[:2]):
                print(f"  {i+1}. {record}")
            
            # 测试保存到CSV
            print("\n测试保存到CSV文件...")
            save_success = save_to_csv(data, "test_acc_list_output.csv")
            if save_success:
                print("CSV文件保存成功: test_acc_list_output.csv")
            else:
                print("CSV文件保存失败")
        
        return True

def main():
    """主函数"""
    print("开始测试光模块API查询工具...")
    
    success = test_api_query()
    
    if success:
        print("\n测试完成!")
    else:
        print("\n测试失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()
