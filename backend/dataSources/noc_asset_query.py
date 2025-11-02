#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOC资产信息查询工具
支持跨平台运行，实现与特定curl命令相同的功能
"""

import requests
import json
import logging
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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
    查询高值耗材信息
    
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

def query_ip_networks(idc_id: int = 1867, start: int = 0, length: int = 1000) -> Dict[str, Any]:
    """
    查询IP网络信息
    
    Args:
        idc_id (int): 机房ID，默认为1867
        start (int): 起始位置，默认为0
        length (int): 数据长度，默认为1000
        
    Returns:
        Dict[str, Any]: API响应结果
    """
    # 请求数据
    payload = {
        "tablename": "network_section",
        "header": "id,name,mask,gateway,parent,belong_type_,belong_,type_,status_id_,isp,charger,usage,remark,idc",
        "start": start,
        "length": length,
        "conditions": {
            "idc_id": idc_id
        }
    }
    
    try:
        logger.info(f"发送IP网络信息API请求，机房ID: {idc_id}...")
        response = requests.post(FULL_URL, headers=HEADERS, data=json.dumps(payload))
        
        # 检查响应状态
        response.raise_for_status()
        
        # 解析JSON响应
        result = response.json()
        logger.info(f"IP网络信息API请求成功，机房ID: {idc_id}，返回 {len(result.get('data', []))} 条记录")
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"IP网络信息API请求失败，机房ID: {idc_id} - {e}")
        return {"error": str(e)}
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败，机房ID: {idc_id} - {e}")
        return {"error": f"JSON解析失败: {e}"}
    except Exception as e:
        logger.error(f"未知错误，机房ID: {idc_id} - {e}")
        return {"error": str(e)}


def fetch_ip_networks_full(idc_id: int = 1867, page_length: int = 1000, max_workers: int = 8) -> List[Dict[str, Any]]:
    """
    并发拉取NOC中IP网络全量数据
    
    Args:
        idc_id (int): 机房ID，默认为1867
        page_length (int): 每页长度，默认为1000
        max_workers (int): 最大并发数，默认为8
        
    Returns:
        List[Dict[str, Any]]: 所有IP网络数据
    """
    all_data = []
    failed_pages = []
    stop_flag = False  # 停止标志
    
    # 定义单页请求函数
    def fetch_page(page_id: int, start: int) -> Dict[str, Any]:
        """获取单页数据"""
        # 检查是否需要停止
        if stop_flag:
            return {"page_id": page_id, "data": None, "error": "Stopped", "start": start}
            
        payload = {
            "tablename": "network_section",
            "header": "id,name,mask,gateway,parent,belong_type_,belong_,type_,status_id_,isp,charger,usage,remark,idc",
            "start": start,
            "length": page_length,
            "conditions": {
                "idc_id": idc_id
            }
        }
        
        try:
            logger.info(f"发送第 {page_id} 页请求，起始位置: {start}")
            response = requests.post(FULL_URL, headers=HEADERS, data=json.dumps(payload), timeout=30)
            response.raise_for_status()
            result = response.json()
            logger.info(f"第 {page_id} 页请求成功，返回 {len(result.get('data', []))} 条记录")
            return {"page_id": page_id, "data": result, "error": None, "start": start}
        except Exception as e:
            logger.error(f"第 {page_id} 页请求失败: {e}")
            return {"page_id": page_id, "data": None, "error": str(e), "start": start}
    
    # 使用线程池并发拉取数据
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        # 提交前1000页的请求（足够覆盖大部分情况）
        max_pages = 1000
        
        # 提交所有页面的请求
        for i in range(max_pages):
            page_id = i + 1
            start = i * page_length
            future = executor.submit(fetch_page, page_id, start)
            futures.append(future)
        
        # 收集结果
        consecutive_empty_pages = 0
        max_consecutive_empty = 3  # 连续3页无数据则停止
        
        for future in as_completed(futures):
            result = future.result()
            page_id = result["page_id"]
            start = result["start"]
            
            # 检查是否被主动停止
            if result["error"] == "Stopped":
                continue
                
            if result["error"]:
                failed_pages.append((page_id, start, result["error"]))
                continue
                
            data = result["data"]
            if not data or "data" not in data:
                logger.info(f"第 {page_id} 页无数据")
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= max_consecutive_empty:
                    logger.info(f"连续 {max_consecutive_empty} 页无数据，停止拉取")
                    stop_flag = True  # 设置停止标志
                    break
                continue
                
            page_data = data["data"]
            if not page_data:
                logger.info(f"第 {page_id} 页无数据")
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= max_consecutive_empty:
                    logger.info(f"连续 {max_consecutive_empty} 页无数据，停止拉取")
                    stop_flag = True  # 设置停止标志
                    break
                continue
                
            # 重置连续空页计数
            consecutive_empty_pages = 0
            all_data.extend(page_data)
            logger.info(f"第 {page_id} 页数据已添加，当前总计 {len(all_data)} 条记录")
    
    # 记录失败的页面
    if failed_pages:
        logger.warning(f"以下页面拉取失败: {failed_pages}")
    
    logger.info(f"IP网络全量数据拉取完成，共 {len(all_data)} 条记录")
    return all_data


def fetch_optical_modules_full(start_id: int = 0, page_length: int = 1000, max_workers: int = 8) -> List[Dict[str, Any]]:
    """
    并发拉取NOC中光模块全量数据，每次ID加1000，每页长度是1000
    
    Args:
        start_id (int): 起始ID，默认为0
        page_length (int): 每页长度，默认为1000
        max_workers (int): 最大并发数，默认为8
        
    Returns:
        List[Dict[str, Any]]: 所有光模块数据
    """
    all_data = []
    failed_pages = []
    stop_flag = False  # 停止标志
    
    # 定义单页请求函数
    def fetch_page(page_id: int, start: int) -> Dict[str, Any]:
        """获取单页数据"""
        # 检查是否需要停止
        if stop_flag:
            return {"page_id": page_id, "data": None, "error": "Stopped", "start": start}
            
        payload = {
            "tablename": "netware_accessory",
            "header": "id,sn,event,type,producer,idc,status,device",
            "start": start,
            "length": page_length,
            "conditions": {
                "status_id": [1, 11],
                "container_id": 498,
                "class_id": 6
            }
        }
        
        try:
            logger.info(f"发送第 {page_id} 页请求，起始ID: {start}")
            response = requests.post(FULL_URL, headers=HEADERS, data=json.dumps(payload), timeout=30)
            response.raise_for_status()
            result = response.json()
            logger.info(f"第 {page_id} 页请求成功，返回 {len(result.get('data', []))} 条记录")
            return {"page_id": page_id, "data": result, "error": None, "start": start}
        except Exception as e:
            logger.error(f"第 {page_id} 页请求失败: {e}")
            return {"page_id": page_id, "data": None, "error": str(e), "start": start}
    
    # 使用线程池并发拉取数据
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        # 提交前1000页的请求（足够覆盖大部分情况）
        max_pages = 1000
        
        # 提交所有页面的请求
        for i in range(max_pages):
            page_id = i + 1
            start = start_id + (i * page_length)
            future = executor.submit(fetch_page, page_id, start)
            futures.append(future)
        
        # 收集结果
        consecutive_empty_pages = 0
        max_consecutive_empty = 3  # 连续3页无数据则停止
        
        for future in as_completed(futures):
            result = future.result()
            page_id = result["page_id"]
            start = result["start"]
            
            # 检查是否被主动停止
            if result["error"] == "Stopped":
                continue
                
            if result["error"]:
                failed_pages.append((page_id, start, result["error"]))
                continue
                
            data = result["data"]
            if not data or "data" not in data:
                logger.info(f"第 {page_id} 页无数据")
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= max_consecutive_empty:
                    logger.info(f"连续 {max_consecutive_empty} 页无数据，停止拉取")
                    stop_flag = True  # 设置停止标志
                    break
                continue
                
            page_data = data["data"]
            if not page_data:
                logger.info(f"第 {page_id} 页无数据")
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= max_consecutive_empty:
                    logger.info(f"连续 {max_consecutive_empty} 页无数据，停止拉取")
                    stop_flag = True  # 设置停止标志
                    break
                continue
                
            # 重置连续空页计数
            consecutive_empty_pages = 0
            all_data.extend(page_data)
            logger.info(f"第 {page_id} 页数据已添加，当前总计 {len(all_data)} 条记录")
    
    # 记录失败的页面
    if failed_pages:
        logger.warning(f"以下页面拉取失败: {failed_pages}")
    
    logger.info(f"光模块全量数据拉取完成，共 {len(all_data)} 条记录")
    return all_data


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
