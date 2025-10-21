#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
光模块库存数据获取
"""

import requests
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple
import re
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from backend.dataSources.cluster_info import get_cluster_info_from_hostname

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API配置
API_URL = "http://10.69.77.238/open-apis/netware/get-acc-info"
ACCESS_TOKEN = "g1erjvg1dmndlkjs3gpmp1483soesjg8"

HEADERS = {
    "Access-Token": ACCESS_TOKEN,
    "Host": "noc.api.sdns.ksyun.com",
    "Content-Type": "application/json",
    "Connection": "keep-alive",
    "Accept": "application/json",
}

PAGE_SIZE = 1000
WORKERS = 8                # 并行处理每页的 worker 数
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0      # 秒
SAFETY_LIMIT = 500000      # 最大允许抓取记录数（去重后）

# 全局并发安全结构
seen_ids = set()           # 用于去重（仅存 id）——在内存中
seen_lock = Lock()         # 锁保护 seen_ids 与 all_count 写入
all_count = 0
all_count_lock = Lock()


def process_module_type(module_type: str) -> str:
    """
    处理光模块型号，去除厂商前缀
    
    Args:
        module_type (str): 原始光模块型号
        
    Returns:
        str: 处理后的光模块型号
    """
    if not module_type:
        return "non-data"
    
    # 去除厂商前缀，如H3C-SFP+500G-SR -> SFP+500G-SR
    parts = module_type.split("-", 1)
    if len(parts) > 1:
        return parts[1]
    return module_type

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

def process_page_data(page_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    对单页数据做处理，返回处理后的数据列表。
    这个函数会被 ThreadPoolExecutor 并行执行以提高吞吐。
    """
    global all_count
    processed_items = []
    added = 0
    # 局部集合以减少锁竞争
    local_new_ids = []
    for item in page_items:
        iid = item.get("id")
        if iid is None:
            # 如果没有 id，可以选择跳过或保存。这里跳过并记录日志
            logger.debug("跳过无 id 的记录")
            continue
        # 使用锁检查并添加到 seen_ids
        with seen_lock:
            if iid in seen_ids:
                continue
            seen_ids.add(iid)
        local_new_ids.append(iid)
        processed_items.append(item)

    added = len(processed_items)
    with all_count_lock:
        all_count += added

    return processed_items

def fetch_all_optical_module_inventory() -> List[Dict[str, Any]]:
    """
    获取所有光模块库存数据（分页获取）
    
    Returns:
        List[Dict[str, Any]]: 处理后的光模块库存数据
    """
    global all_count, seen_ids
    # 重置全局变量
    seen_ids.clear()
    all_count = 0
    
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

    # 线程池用于并行处理每页
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = []  # 保存正在处理的 Future

        while True:
            page_no += 1
            payload = {
                "idc": "",
                "producer": "",
                "container": "业务部门网络零件",
                "class": "光模块",
                "status": ["在线", "流程处理中"],
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

            acc_list = result.get("data", {}).get("acc_list", []) or []
            if not acc_list:
                logger.info(f"第 {page_no} 页无数据，结束")
                break

            # 按你确认的规则：使用返回 json 的最后一条记录的 id 作为下一次 min_id
            last_rec = acc_list[-1]
            last_id = last_rec.get("id")
            if last_id is None:
                logger.error(f"第 {page_no} 页最后一条记录无 id，停止抓取以避免死循环")
                failed_pages.append((page_no, "last_id_missing"))
                break

            # 把当前页的 acc_list 提交给线程池并行处理
            future = pool.submit(process_page_data, acc_list)
            futures.append((page_no, future))

            # 更新 min_id（按你确认的逻辑直接设置为返回的最后一条 id）
            min_id = last_id

            # 进度打印：统计已去重计数（注意处理可能仍在进行中）
            with all_count_lock:
                current_count = all_count
            logger.info(f"已请求页 {page_no}，返回 {len(acc_list)} 条，累计已去重 {current_count} 条 (reported total={page_total})")

            # 安全上限检查（去重后）
            if current_count >= SAFETY_LIMIT:
                logger.warning(f"到达安全上限 {SAFETY_LIMIT} 条，停止抓取")
                break

            # 小睡以防某些网关限制频率
            time.sleep(0.01)

        # 等待所有提交的处理任务完成（或超时/取消）
        logger.info("等待并行处理任务完成...")
        for page_idx, fut in futures:
            try:
                page_data = fut.result(timeout=300)  # 可根据需要调整超时时间
                all_data.extend(page_data)
                logger.debug(f"page {page_idx} 处理完成，新增 {len(page_data)} 条")
            except Exception as e:
                logger.exception(f"page {page_idx} 处理异常: {e!r}")
                failed_pages.append((page_idx, f"process_error:{e!r}"))

    # 最终统计
    with all_count_lock:
        final_count = all_count
    logger.info(f"抓取结束。去重后总记录数: {final_count}")
    if total_reported is not None:
        logger.info(f"最后一次 page_total 报告为: {total_reported}")
    if failed_pages:
        logger.warning(f"部分页失败/异常: {failed_pages}")

    # 处理数据
    processed_data = []
    # 用于聚合相同记录的字典
    aggregated_data = {}
    
    # 获取当前时间作为日期字段
    current_time = datetime.now().isoformat()
    
    for item in all_data:
        # 提取字段
        idc = item.get("idc", "non-data")
        module_type = item.get("type", "non-data")
        producer = item.get("producer", "non-data")
        hostname = item.get("hostname", "non-data")
        
        # 处理光模块型号
        processed_module_type = process_module_type(module_type)
        
        # 获取集群和客户信息
        cluster, customer = get_cluster_info_from_hostname(hostname,is_roce_event=False)
        
        # 创建聚合键
        key = (processed_module_type, producer, idc, cluster, customer)
        if key in aggregated_data:
            aggregated_data[key] += 1
        else:
            aggregated_data[key] = 1
    
    # 转换聚合数据为列表
    for (module_type, producer, idc, cluster, customer), count in aggregated_data.items():
        processed_data.append({
            "日期": current_time,
            "光模块型号": module_type,
            "光模块厂商": producer,
            "机房": idc,
            "集群": cluster,
            "客户": customer,
            "在线总数量": count
        })
    
    logger.info(f"处理完成，共生成 {len(processed_data)} 条记录")
    return processed_data

def query_optical_module_inventory():
    """
    查询光模块库存数据（供数据加载器调用）
    
    Returns:
        List[Dict[str, Any]]: 处理后的光模块库存数据
    """
    return fetch_all_optical_module_inventory()

def main():
    """主函数"""
    try:
        data = query_optical_module_inventory()
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
