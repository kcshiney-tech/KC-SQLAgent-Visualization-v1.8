#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
光模块API查询工具
支持跨平台运行，实现与特定curl命令相同的功能
"""

import requests
import json
import logging
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from typing import Dict, Any, List

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API配置
API_URL = "https://noc.ksyun.com/api/get-list"
ACCESS_TOKEN = "g4m5o9w8m7n2b8d2l5p1w7f4p2r9y9q2"
FULL_URL = f"{API_URL}?access_token={ACCESS_TOKEN}"

HEADERS = {
    "Content-Type": "application/json"
}

def query_optical_modules(batch_query: str) -> Dict[str, Any]:
    """
    查询光模块信息
    
    Args:
        batch_query (str): 批量查询的序列号，用换行符分隔
        
    Returns:
        Dict[str, Any]: API响应结果
    """
    # 请求数据
    payload = {
        "tablename": "netware_accessory",
        "header": "id,sn,event,class,type,producer,idc,status,device,list",
        "status_id": 1,
        "limit": -1,
        "conditions": {
            "batch_query": batch_query
        }
    }
    
    try:
        logger.info("发送API请求...")
        response = requests.post(FULL_URL, headers=HEADERS, data=json.dumps(payload))
        
        # 检查响应状态
        response.raise_for_status()
        
        # 解析JSON响应
        result = response.json()
        logger.info(f"API请求成功，返回 {len(result.get('data', []))} 条记录")
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API请求失败: {e}")
        return {"error": str(e)}
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {e}")
        return {"error": f"JSON解析失败: {e}"}
    except Exception as e:
        logger.error(f"未知错误: {e}")
        return {"error": str(e)}

def save_to_csv(data: List[Dict[str, Any]], filename: str = "acc_list_output2.csv") -> bool:
    """
    将数据保存为CSV文件
    
    Args:
        data (List[Dict[str, Any]]): 要保存的数据
        filename (str): 输出文件名
        
    Returns:
        bool: 保存是否成功
    """
    try:
        if not data:
            logger.warning("没有数据可保存")
            return False
            
        # 获取所有可能的列名
        fieldnames = set()
        for item in data:
            fieldnames.update(item.keys())
        fieldnames = sorted(list(fieldnames))
        
        # 写入CSV文件
        import csv
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
            
        logger.info(f"数据已保存到 {filename}")
        return True
        
    except Exception as e:
        logger.error(f"保存CSV文件失败: {e}")
        return False

def main():
    """主函数"""
    # 默认的批量查询内容，可根据需要修改
    batch_query_content = """CVJH05023511277 
210231A562N144011744"""
    
    print("光模块API查询工具")
    print("=" * 30)
    print("默认查询序列号:")
    print(batch_query_content)
    print("=" * 30)
    
    # 执行查询
    result = query_optical_modules(batch_query_content)
    
    # 检查是否有错误
    if "error" in result:
        print(f"查询失败: {result['error']}")
        return
    
    # 显示结果
    data = result.get("data", [])
    print(f"查询成功，共获得 {len(data)} 条记录")
    
    # 显示前几条记录
    if data:
        print("\n前3条记录:")
        for i, record in enumerate(data[:3]):
            print(f"  {i+1}. {record}")
    
    # 保存到CSV文件
    if data:
        save_success = save_to_csv(data, "acc_list_output2.csv")
        if save_success:
            print("\n数据已保存到 acc_list_output2.csv")
        else:
            print("\n数据保存失败")

if __name__ == "__main__":
    main()
