#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络设备故障事件查询
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
from backend.dataSources.mongodb_connector import connect_mongodb, get_database
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_cluster_info_from_hostname(hostname: str) -> tuple:
    """
    根据主机名确定集群名和客户名称
    只有ROCE/SROCE_TOR/AGG/CORE-0x是有集群和客户的
    
    Args:
        hostname (str): 主机名
        
    Returns:
        tuple: (cluster_name, customer_name)
    """
    if not hostname or hostname == "non-data":
        return ("", "")
    
    # 已知的集群映射
    cluster_mapping = {
        "QH-QHDX-AZ-ROCE_TOR-02": ("QHDX02", "小米"),
        "NB-LT-AZ-ROCE_TOR-01": ("NBLT01", "百川"),
        "QY-YD-DC-ROCE_TOR-05": ("QYYD05", "月暗"),
        "QY-ZNJ-DC-ROCE_TOR-01": ("QYZNJ01", "云启")
    }
    
    # 检查是否匹配已知的集群
    for prefix, (cluster_name, customer_name) in cluster_mapping.items():
        if hostname.startswith(prefix):
            # 处理重复字符的情况，如QHQHYD0x -> QHYD0x
            if len(cluster_name) >= 6 and cluster_name[:2] == cluster_name[2:4]:
                cluster_name = cluster_name[2:]
            
            # 根据主机名添加前缀
            if "-ROCE_TOR-" in hostname:
                cluster_name = f"ROCE-TOR-{cluster_name}"
            elif "-SROCE_TOR-" in hostname:
                cluster_name = f"SROCE-TOR-{cluster_name}"
            elif "-ROCE_AGG-" in hostname:
                cluster_name = f"ROCE-AGG-{cluster_name}"
            elif "-SROCE_AGG-" in hostname:
                cluster_name = f"SROCE-AGG-{cluster_name}"
            elif "-ROCE_CORE-" in hostname:
                cluster_name = f"ROCE-CORE-{cluster_name}"
            elif "-SROCE_CORE-" in hostname:
                cluster_name = f"SROCE-CORE-{cluster_name}"
                
            return (cluster_name, customer_name)
    
    # 检查是否匹配ROCE_TOR模式
    tor_match = re.match(r'^[A-Z0-9-]*-ROCE_TOR-\d+', hostname)
    if tor_match:
        # 提取机房信息，从主机名前面两个字符串拼接在一起
        parts = hostname.split('-')
        if len(parts) >= 2:
            # 如果前两个部分相同，使用第二个部分；否则使用前两个部分拼接
            if parts[0] == parts[1]:
                idc = parts[0]
            else:
                idc = parts[0] + parts[1]
            # 查找数字部分
            num_match = re.search(r'-ROCE_TOR-(\d+)', hostname)
            if num_match:
                number = num_match.group(1)
                cluster_name = f"ROCE-TOR-{idc}0{number[-1] if number else 'x'}"
                return (cluster_name, "")
    
    # 检查是否匹配SROCE_TOR模式
    stor_match = re.match(r'^[A-Z0-9-]*-SROCE_TOR-\d+', hostname)
    if stor_match:
        # 提取机房信息，从主机名前面两个字符串拼接在一起
        parts = hostname.split('-')
        if len(parts) >= 2:
            # 如果前两个部分相同，使用第二个部分；否则使用前两个部分拼接
            if parts[0] == parts[1]:
                idc = parts[0]
            else:
                idc = parts[0] + parts[1]
            # 查找数字部分
            num_match = re.search(r'-SROCE_TOR-(\d+)', hostname)
            if num_match:
                number = num_match.group(1)
                cluster_name = f"SROCE-TOR-{idc}0{number[-1] if number else 'x'}"
                return (cluster_name, "")
    
    # 检查是否匹配ROCE_AGG模式
    agg_match = re.match(r'^[A-Z0-9-]*-ROCE_AGG-\d+', hostname)
    if agg_match:
        # 提取机房信息，从主机名前面两个字符串拼接在一起
        parts = hostname.split('-')
        if len(parts) >= 2:
            # 如果前两个部分相同，使用第二个部分；否则使用前两个部分拼接
            if parts[0] == parts[1]:
                idc = parts[0]
            else:
                idc = parts[0] + parts[1]
            # 查找数字部分
            num_match = re.search(r'-ROCE_AGG-(\d+)', hostname)
            if num_match:
                number = num_match.group(1)
                cluster_name = f"ROCE-AGG-{idc}0{number[-1] if number else 'x'}"
                return (cluster_name, "")
    
    # 检查是否匹配SROCE_AGG模式
    sagg_match = re.match(r'^[A-Z0-9-]*-SROCE_AGG-\d+', hostname)
    if sagg_match:
        # 提取机房信息，从主机名前面两个字符串拼接在一起
        parts = hostname.split('-')
        if len(parts) >= 2:
            # 如果前两个部分相同，使用第二个部分；否则使用前两个部分拼接
            if parts[0] == parts[1]:
                idc = parts[0]
            else:
                idc = parts[0] + parts[1]
            # 查找数字部分
            num_match = re.search(r'-SROCE_AGG-(\d+)', hostname)
            if num_match:
                number = num_match.group(1)
                cluster_name = f"SROCE-AGG-{idc}0{number[-1] if number else 'x'}"
                return (cluster_name, "")
    
    # 检查是否匹配CORE模式（ROCE_CORE）
    core_match = re.match(r'^[A-Z0-9-]*-ROCE_CORE-\d+', hostname)
    if core_match:
        # 提取机房信息，从主机名前面两个字符串拼接在一起
        parts = hostname.split('-')
        if len(parts) >= 2:
            # 如果前两个部分相同，使用第二个部分；否则使用前两个部分拼接
            if parts[0] == parts[1]:
                idc = parts[0]
            else:
                idc = parts[0] + parts[1]
            # 查找数字部分
            num_match = re.search(r'-ROCE_CORE-(\d+)', hostname)
            if num_match:
                number = num_match.group(1)
                cluster_name = f"ROCE-CORE-{idc}0{number[-1] if number else 'x'}"
                return (cluster_name, "")
    
    # 检查是否匹配SCORE模式（SROCE_CORE）
    score_match = re.match(r'^[A-Z0-9-]*-SROCE_CORE-\d+', hostname)
    if score_match:
        # 提取机房信息，从主机名前面两个字符串拼接在一起
        parts = hostname.split('-')
        if len(parts) >= 2:
            # 如果前两个部分相同，使用第二个部分；否则使用前两个部分拼接
            if parts[0] == parts[1]:
                idc = parts[0]
            else:
                idc = parts[0] + parts[1]
            # 查找数字部分
            num_match = re.search(r'-SROCE_CORE-(\d+)', hostname)
            if num_match:
                number = num_match.group(1)
                cluster_name = f"SROCE-CORE-{idc}0{number[-1] if number else 'x'}"
                return (cluster_name, "")
    
    # 没匹配上集群名的，集群和客户处为空
    return ("", "")


# ==========================================
# 网络设备故障事件查询
# ==========================================
def query_network_device_failure():
    # 数据库连接配置
    event_monitor_mongo_host = "10.69.74.235"
    event_monitor_mongo_port  = 8526
    event_monitor_mongo_db    = "network_event"
    event_monitor_mongo_user  = "network_event_ro_test"
    event_monitor_mongo_pass  = "Ro123456"

    # 连接数据库
    if not connect_mongodb(host=event_monitor_mongo_host, port=event_monitor_mongo_port,
                          database=event_monitor_mongo_db,
                          username=event_monitor_mongo_user, password=event_monitor_mongo_pass):
        logger.error("事件监控数据库连接失败")
        return []

    # 通用查询方法（简化版）
    def query_event_monitor_event_list(
        event_types: List[str],
        is_finished: Optional[bool] = True,    # True表示已完成
        is_one_click_finish: Optional[bool] = False,    # False表示非一键完成
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        database: str = "network_event",
        collection: str = "event"
    ) -> Dict[str, Any]:
        try:
            db = get_database(database)
            if db is None:
                logger.error("数据库未连接")
                return {"total": 0, "data": []}

            coll = db[collection]
            query = {}

            # 查询多种事件类型
            if event_types:
                query["event_type"] = {"$in": event_types}

            if is_finished is not None:
                if is_finished:
                    query["status_code"] = 100
                else:
                    query["status_code"] = {"$lt": 100}

            if is_one_click_finish is not None:
                if is_one_click_finish:
                    query["is_one_click_finish"] = "一键完成"
                else:
                    query["is_one_click_finish"] = {"$ne": "一键完成"}

            if start_time and end_time:
                query["starts_at"] = {"$gte": start_time, "$lte": end_time}
            elif start_time:
                query["starts_at"] = {"$gte": start_time}
            elif end_time:
                query["starts_at"] = {"$lte": end_time}

            total = coll.count_documents(query)
            cursor = coll.find(query).sort("starts_at", -1).skip(offset)

            if limit is not None:
                cursor = cursor.limit(limit)

            results = list(cursor)
            logger.info(f"查询到 {total} 条记录，返回 {len(results)} 条")
            return {"total": total, "data": results}

        except Exception as e:
            logger.error(f"查询失败: {e}")
            return {"total": 0, "data": []}

    # 集中设置时间范围
    days = 90  # 可以根据需要调整
    end = datetime.now()
    start = (end - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    logger.info(f"查询时间范围: {start} 至 {end}")

    # 分别查询两种类型的事件以调试
    event_types = ["NetworkDevice", "EventTypeNetworkDeviceBoard"]
    is_finished = None  # True表示只查询已结束的事件
    is_one_click_finish = None  # False表示只查询非一键完成的事件

    result = query_event_monitor_event_list(
        event_types=event_types,
        is_finished=is_finished,
        is_one_click_finish=is_one_click_finish,
        start_time=start,
        end_time=end,
        limit=None  # 不限制数量
    )

    # 调试信息：统计不同事件类型的数量
    event_type_count = {}
    for event in result["data"]:
        event_type = event.get("event_type", "unknown")
        event_type_count[event_type] = event_type_count.get(event_type, 0) + 1
    
    logger.info(f"事件类型分布: {event_type_count}")

    # 处理数据
    processed_data = []
    
    for event in result["data"]:
        # 提取事件相关信息
        event_id = event.get("event_id", "non-data")
        event_type = event.get("event_type", "non-data")
        event_name = event.get("event_name", "non-data")
        hostname = event.get("hostname", "non-data")
        device_idc = event.get("device_idc", "non-data")
        device_slot = event.get("device_slot", "non-data")
        description = event.get("description", "non-data")
        
        # 提取时间字段
        starts_at = event.get("starts_at")
        claimed_at = event.get("claimed_at")
        case_at = event.get("case_at")
        ends_at = event.get("ends_at")
        
        # 格式化时间字段
        def format_datetime(dt):
            if dt:
                if isinstance(dt, str):
                    try:
                        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
                    except:
                        return "non-data"
                return dt.isoformat()
            return "non-data"
        
        starts_at_str = format_datetime(starts_at)
        claimed_at_str = format_datetime(claimed_at)
        case_at_str = format_datetime(case_at)
        ends_at_str = format_datetime(ends_at)
        
        # 获取NOC工单信息
        noc_case = event.get("noc_case")
        if noc_case is None:
            noc_case = {}
        noc_case_id = noc_case.get("noc_case_id", "non-data")
        outsource_ids = noc_case.get("outsource_ids", "non-data")
        fault_type = noc_case.get("fault_type", "non-data")
        
        # 获取集群和客户信息
        cluster, customer = get_cluster_info_from_hostname(hostname)
        
        # 创建记录
        processed_data.append({
            "事件ID": event_id,
            "事件类型": event_type,
            "事件名称": event_name,
            "设备主机名": hostname,
            "机房": device_idc,
            "集群": cluster,
            "客户": customer,
            "设备槽位": device_slot,
            "描述": description,
            "开始时间": starts_at_str,
            "认领时间": claimed_at_str,
            "建单时间": case_at_str,
            "结束时间": ends_at_str,
            "NOC工单ID": noc_case_id,
            "外包工单ID": outsource_ids,
            "工单故障类型": fault_type
        })
    
    logger.info(f"处理完成，共生成 {len(processed_data)} 条记录")
    return processed_data


def main():
    try:
        # 网络设备故障事件查询
        data = query_network_device_failure()
        if data:
            print(f"成功获取 {len(data)} 条记录")
            print("前5条记录:")
            for i, record in enumerate(data[:5]):
                print(f"  {i+1}. {record}")
        else:
            print("未能获取数据")
    except Exception as e:
        logger.error(f"执行查询时出错: {e}")
        print(f"错误: {e}")

if __name__ == "__main__":
    main()
