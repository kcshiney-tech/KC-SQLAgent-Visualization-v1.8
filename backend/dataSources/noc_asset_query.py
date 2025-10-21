#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOC资产信息查询工具
支持跨平台运行，实现与特定curl命令相同的功能
"""

import requests
import json
import logging
from typing import Dict, Any

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

def query_noc_assets(tablename: str, header: str, batch_query: str, status_id: int = 1, limit: int = -1) -> Dict[str, Any]:
    """
    查询NOC资产信息
    
    Args:
        tablename (str): 表名
        header (str): 列列表，用逗号分隔
        batch_query (str): 批量查询的序列号，用换行符分隔
        status_id (int): 状态ID，默认为1
        limit (int): 限制返回记录数，默认为-1（不限制）
        
    Returns:
        Dict[str, Any]: API响应结果
    """
    # 请求数据
    payload = {
        "tablename": tablename,
        "header": header,
        "status_id": status_id,
        "limit": limit,
        "conditions": {
            "batch_query": batch_query
        }
    }
    
    try:
        logger.info(f"发送NOC资产API请求: {tablename}...")
        response = requests.post(FULL_URL, headers=HEADERS, data=json.dumps(payload))
        
        # 检查响应状态
        response.raise_for_status()
        
        # 解析JSON响应
        result = response.json()
        logger.info(f"NOC资产API请求成功: {tablename}，返回 {len(result.get('data', []))} 条记录")
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"NOC资产API请求失败: {tablename} - {e}")
        return {"error": str(e)}
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {tablename} - {e}")
        return {"error": f"JSON解析失败: {e}"}
    except Exception as e:
        logger.error(f"未知错误: {tablename} - {e}")
        return {"error": str(e)}

def query_optical_modules(batch_query: str) -> Dict[str, Any]:
    """
    查询光模块信息
    
    Args:
        batch_query (str): 批量查询的序列号，用换行符分隔
        
    Returns:
        Dict[str, Any]: API响应结果
        
    Header columns explanation:
        id: ID
        sn: 序列号
        event: 事件
        class: 类别
        type: 型号
        producer: 生产商
        idc: 机房
        status: 状态
        device: 设备
        list: 列表
    """
    return query_noc_assets(
        tablename="netware_accessory",
        header="id,sn,event,class,type,producer,idc,status,device,list",
        batch_query=batch_query
    )

def query_cable_assets(batch_query: str) -> Dict[str, Any]:
    """
    查询电缆资产信息
    
    Args:
        batch_query (str): 批量查询的序列号，用换行符分隔
        
    Returns:
        Dict[str, Any]: API响应结果
        
    Header columns explanation:
        id: ID
        idc_id: 机房ID
        type: 类型
        model: 型号
        sn_a: A端序列号
        sn_b: B端序列号
        length: 长度
        brand: 品牌
        status: 状态
        maintenance_time: 维护时间
    """
    return query_noc_assets(
        tablename="cable_high_value_asset",
        header="id,idc_id,type,model,sn_a,sn_b,length,brand,status,maintenance_time",
        batch_query=batch_query
    )

def main():
    """主函数"""
    # 默认的批量查询内容，可根据需要修改
    batch_query_content = """CVJH05023511277 
210231A562N144011744"""
    
    print("NOC资产API查询工具")
    print("=" * 30)
    print("默认查询序列号:")
    print(batch_query_content)
    print("=" * 30)
    
    # 执行光模块查询
    print("\n查询光模块信息:")
    result = query_optical_modules(batch_query_content)
    
    # 检查是否有错误
    if "error" in result:
        print(f"查询失败: {result['error']}")
    else:
        # 显示结果
        data = result.get("data", [])
        print(f"查询成功，共获得 {len(data)} 条记录")
        
        # 显示前几条记录
        if data:
            print("\n前3条记录:")
            for i, record in enumerate(data[:3]):
                print(f"  {i+1}. {record}")

if __name__ == "__main__":
    main()
