#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络设备数据获取
"""

import requests
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple
import json
import time
import re

from backend.dataSources.cluster_info import get_cluster_info_from_hostname

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API配置
API_URL = "http://10.69.77.238/open-apis/netware/get-device-info"
ACCESS_TOKEN = "g1erjvg1dmndlkjs3gpmp1483soesjg8"

HEADERS = {
    "Access-Token": ACCESS_TOKEN,
    "Host": "noc.api.sdns.ksyun.com",
    "Content-Type": "application/json",
    "Connection": "keep-alive",
    "Accept": "application/json",
}

PAGE_SIZE = 1000
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0      # 秒
SAFETY_LIMIT = 500000      # 最大允许抓取记录数（去重后）


def backoff_sleep(attempt: int):
    # 指数退避，带少量抖动
    delay = INITIAL_BACKOFF * (2 ** (attempt - 1))
    jitter = delay * 0.1
    time.sleep(delay + (jitter * (0.5 - time.time() % 1)))  # 简单伪抖动

def safe_post(session: requests.Session, payload: Dict[str, Any], timeout: int = 30) -> requests.Response:
    """带重试的 POST（在请求层面捕获 5xx/连接错误），不捕获 4xx (视为逻辑错误)"""
    last_exc = RuntimeError("Unknown error")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.post(API_URL, json=payload, timeout=timeout)
            # 如果是 5xx 或 502，触发重试
            if 500 <= resp.status_code < 600:
                logger.warning(f"请求返回 {resp.status_code}，第 {attempt} 次重试")
                last_exc = RuntimeError(f"HTTP {resp.status_code}")
                backoff_sleep(attempt)
                continue
            return resp
        except requests.RequestException as e:
            last_exc = e
            logger.warning(f"请求异常（第 {attempt} 次）: {e!r}")
            backoff_sleep(attempt)
            continue
    # 全部重试失败，抛出最后一次异常
    logger.error(f"请求重试 {MAX_RETRIES} 次失败")
    raise last_exc

def fetch_all_network_device_inventory() -> List[Dict[str, Any]]:
    """
    获取所有网络设备库存数据（分页获取）
    
    Returns:
        List[Dict[str, Any]]: 处理后的网络设备库存数据
    """
    session = requests.Session()
    session.trust_env = False
    session.headers.update(HEADERS)

    min_id = 0
    total_reported = None
    page_no = 0
    # 记录失败页信息
    failed_pages = []
    
    # 存储所有获取到的数据
    all_data = []

    while True:
        page_no += 1
        payload = {
            "idc": "",
            "producer": "",
            "status": ["在线"],
            "page_size": PAGE_SIZE,
            "min_id": min_id
        }

        try:
            resp = safe_post(session, payload, timeout=30)
        except Exception as e:
            logger.exception(f"第 {page_no} 页请求失败，退出抓取: {e!r}")
            failed_pages.append((page_no, str(e)))
            break

        # 若非200，记录并根据需要退出
        if resp.status_code != 200:
            logger.error(f"第 {page_no} 页返回 HTTP {resp.status_code}：{resp.text[:200]}")
            failed_pages.append((page_no, f"HTTP {resp.status_code}"))
            break

        try:
            result = resp.json()
        except Exception as e:
            logger.exception(f"第 {page_no} 页 JSON 解析失败，响应头: {resp.headers}")
            failed_pages.append((page_no, "json_parse_error"))
            break

        # 解析 total（后续页可能变化）
        page_total = result.get("data", {}).get("total")
        if total_reported is None:
            total_reported = page_total
            logger.info(f"首次返回 total = {total_reported}")
        else:
            if page_total != total_reported:
                logger.info(f"第 {page_no} 页返回 total = {page_total}（变化，之前={total_reported}）")
                total_reported = page_total

        device_list = result.get("data", {}).get("device_list", []) or []
        if not device_list:
            logger.info(f"第 {page_no} 页无数据，结束")
            break

        # 按你确认的规则：使用返回 json 的最后一条记录的 id 作为下一次 min_id
        last_rec = device_list[-1]
        last_id = last_rec.get("id")
        if last_id is None:
            logger.error(f"第 {page_no} 页最后一条记录无 id，停止抓取以避免死循环")
            failed_pages.append((page_no, "last_id_missing"))
            break

        # 添加当前页数据到总数据中
        all_data.extend(device_list)

        # 更新 min_id（按你确认的逻辑直接设置为返回的最后一条 id）
        min_id = last_id

        logger.info(f"已请求页 {page_no}，返回 {len(device_list)} 条，累计 {len(all_data)} 条 (reported total={page_total})")

        # 安全上限检查
        if len(all_data) >= SAFETY_LIMIT:
            logger.warning(f"到达安全上限 {SAFETY_LIMIT} 条，停止抓取")
            break

        # 小睡以防某些网关限制频率
        time.sleep(0.01)

    # 最终统计
    logger.info(f"抓取结束。总记录数: {len(all_data)}")
    if total_reported is not None:
        logger.info(f"最后一次 page_total 报告为: {total_reported}")
    if failed_pages:
        logger.warning(f"部分页失败/异常: {failed_pages}")

    # 处理数据
    processed_data = []
    # 获取当前时间作为更新时间
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for item in all_data:
        # 提取字段
        hostname = item.get("hostname", "non-data")
        producer = item.get("producer", "non-data")
        device_type = item.get("type", "non-data")
        sn = item.get("sn", "non-data")
        idc = item.get("idc", "non-data")
        rack = item.get("rack", "non-data")
        rack_ip = item.get("rack_ip", "non-data")
        status = item.get("status", "non-data")
        is_maintenance = item.get("is_maintenance", "non-data")
        maintenance_end_time = item.get("maintenance_end_time", "non-data")
        
        # 获取集群和客户信息
        cluster, customer = get_cluster_info_from_hostname(hostname,is_roce_event=False)
        
        processed_data.append({
            "设备主机名": hostname,
            "厂商": producer,
            "型号": device_type,
            "设备SN": sn,
            "机房": idc,
            "机柜位": rack,
            "管理IP": rack_ip,
            "集群": cluster,
            "客户": customer,
            "状态": status,
            "是否过保": is_maintenance,
            "维保到期时间": maintenance_end_time,
            "更新时间": current_time
        })
    
    logger.info(f"处理完成，共生成 {len(processed_data)} 条记录")
    return processed_data

def query_network_device_inventory():
    """
    查询网络设备库存数据（供数据加载器调用）
    
    Returns:
        List[Dict[str, Any]]: 处理后的网络设备库存数据
    """
    return fetch_all_network_device_inventory()

def main():
    """主函数"""
    try:
        data = query_network_device_inventory()
        if data:
            print(f"成功获取 {len(data)} 条记录")
            print("前5条记录:")
            for i, record in enumerate(data[:5]):
                print(f"  {i+1}. {record}")
        else:
            print("未能获取数据")
    except Exception as e:
        logger.error(f"执行时出错: {e}")
        print(f"错误: {e}")

if __name__ == "__main__":
    main()
