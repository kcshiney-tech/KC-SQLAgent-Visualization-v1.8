#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件查询
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
from backend.dataSources.mongodb_connector import connect_mongodb, get_database
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 集群映射函数
def get_cluster_info(hostname: str, is_tor: bool = False, is_agg: bool = False) -> tuple:
    """
    根据主机名确定集群名和客户名称
    
    Args:
        hostname (str): 主机名
        is_tor (bool): 是否为TOR设备
        is_agg (bool): 是否为AGG设备
        
    Returns:
        tuple: (cluster_name, customer_name)
    """
    if not hostname or hostname == "no-data":
        return ("", "")  # 没匹配上集群名的，集群和客户处为空
    
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
            
            # 根据设备类型添加前缀
            if is_tor:
                cluster_name = f"ROCE-TOR上联-{cluster_name}"
            elif is_agg:
                cluster_name = f"ROCE-AGG下联-{cluster_name}"
                
            return (cluster_name, customer_name)
    
    # 对于其他主机名，提取前两个部分作为集群名
    # 例如: NB-LT-AZ-ROCE_TOR-01-xxx -> NBLT01
    tor_match = re.match(r'^([A-Z]+)-([A-Z0-9]+)-[A-Z0-9-]*ROCE_TOR-(\d+)', hostname)
    agg_match = re.match(r'^([A-Z]+)-([A-Z0-9]+)-[A-Z0-9-]*ROCE_AGG-(\d+)', hostname)
    
    if tor_match:
        part1, part2, number = tor_match.groups()
        cluster_name = f"{part1}{part2}0{number[-1]}"
        # 处理重复字符的情况
        if len(cluster_name) >= 6 and cluster_name[:2] == cluster_name[2:4]:
            cluster_name = cluster_name[2:]
        # TOR设备添加"ROCE-TOR上联-"前缀
        if is_tor:
            cluster_name = f"ROCE-TOR上联-{cluster_name}"
        return (cluster_name, "no-data")
    elif agg_match:
        part1, part2, number = agg_match.groups()
        cluster_name = f"{part1}{part2}0{number[-1]}"
        # 处理重复字符的情况
        if len(cluster_name) >= 6 and cluster_name[:2] == cluster_name[2:4]:
            cluster_name = cluster_name[2:]
        # AGG设备添加"ROCE-AGG下联-"前缀
        if is_agg:
            cluster_name = f"ROCE-AGG下联-{cluster_name}"
        return (cluster_name, "no-data")
    
    # 没匹配上集群名的，集群和客户处为空
    return ("", "")


# ==========================================
# 基础网络事件查询
# ==========================================
def query_basicnet_event_demo():
    # 数据库连接配置
    # basicnet_event_mongo_host = "10.69.74.235"
    # basicnet_event_mongo_port  = 8526
    # basicnet_event_mongo_db    = "alertpolicy"
    # basicnet_event_mongo_user  = "network_event_ro_user_sysnet"
    # basicnet_event_mongo_pass  = "rosysnet5678"
    basicnet_event_mongo_host = "10.69.74.235"
    basicnet_event_mongo_port  = 8526
    basicnet_event_mongo_db    = "alertpolicy"
    basicnet_event_mongo_user  = "network_event_ro_user_sysnet"
    basicnet_event_mongo_pass  = "rosysnet5678"

    # 连接数据库
    if not connect_mongodb(host=basicnet_event_mongo_host, port=basicnet_event_mongo_port,
                          database=basicnet_event_mongo_db,
                          username=basicnet_event_mongo_user, password=basicnet_event_mongo_pass):
        logger.error("基础网络事件数据库连接失败")
        return

    # 查询
    def query_basicnet_event_event_list(
        event_code: Optional[str] = None,
        is_finished: Optional[bool] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        database: str = "alertpolicy",
        collection: str = "global_network_event"
    ) -> Dict[str, Any]:
        try:
            db = get_database(database)
            if db is None:
                logger.error("数据库未连接")
                return {"total": 0, "data": []}

            coll = db[collection]
            query = {}

            if event_code:
                query["event_code"] = event_code
            else:
                query["event_code"] = {"$nin": ["POWER_ALARM"]}

            if is_finished is not None:
                if is_finished:
                    query["process_code"] = 40
                else:
                    query["process_code"] = {"$lt": 40}

            if start_time and end_time:
                start_ts = int(start_time.timestamp())
                end_ts = int(end_time.timestamp())
                query["create_ts"] = {"$gte": start_ts, "$lte": end_ts}
            elif start_time:
                start_ts = int(start_time.timestamp())
                query["create_ts"] = {"$gte": start_ts}
            elif end_time:
                end_ts = int(end_time.timestamp())
                query["create_ts"] = {"$lte": end_ts}

            total = coll.count_documents(query)
            cursor = coll.find(query).sort("create_ts", -1).skip(offset)

            if limit is not None:
                cursor = cursor.limit(limit)

            results = list(cursor)
            logger.info(f"查询到 {total} 条记录，返回 {len(results)} 条")
            return {"total": total, "data": results}

        except Exception as e:
            logger.error(f"查询失败: {e}")
            return {"total": 0, "data": []}

    event_code = None  # None表示查询所有事件类型
    start = None
    end = None
    is_finished = True  # True表示只查询已完成的事件, False表示只查询未完成的事件

    result = query_basicnet_event_event_list(
        event_code=event_code,
        is_finished=is_finished,
        start_time=start,
        end_time=end,
        limit=None  # 不限制数量
    )

    for event in result["data"]:
        # 判断是否一键完成
        is_quick_finish = "否"
        if event.get("process_record"):
            record = event.get("process_record",[])[-1]
            if record.get("step") == "一键批量完结":
                is_quick_finish = "是"

        print("事件ID:", event.get("event_id"),
              "事件类型:", event.get("event_name"),
              "开始时间:", event.get("create_time"),
              "是否一键完成:", is_quick_finish)

        for i, r in enumerate(event.get("devices", [])):
            print(f"  设备 {i}: 设备名: {r.get('hostname')}, 端口: {r.get('portname')}, "
                  f"光模块厂商: {r.get('optical_module_vendor')}, 光模块型号: {r.get('optical_module_name')}, "
                  f"光模块SN: {r.get('optical_module_sn')}")
        print()

    print("总记录数:", result["total"])

# ==========================================
# 事件监控查询
# ==========================================
def query_event_monitor_demo():
    # 数据库连接配置
    event_monitor_mongo_host = "10.69.74.235"
    event_monitor_mongo_port  = 8526
    event_monitor_mongo_db    = "network_event"
    event_monitor_mongo_user  = "network_event_ro_user_sysnet"
    event_monitor_mongo_pass  = "rosysnet5678"

    # 连接数据库
    if not connect_mongodb(host=event_monitor_mongo_host, port=event_monitor_mongo_port,
                          database=event_monitor_mongo_db,
                          username=event_monitor_mongo_user, password=event_monitor_mongo_pass):
        logger.error("事件监控数据库连接失败")
        return []

    # 通用查询方法（简化版）
    def query_event_monitor_event_list(
        event_type: Optional[str] = None,
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

            if event_type:
                query["event_type"] = event_type

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

    event_type = "NetworkDeviceInterface"
    is_finished = True  # True表示只查询已结束的事件
    is_one_click_finish = False  # False表示只查询非一键完成的事件

    result = query_event_monitor_event_list(
        event_type=event_type,
        is_finished=is_finished,
        is_one_click_finish=is_one_click_finish,
        start_time=start,
        end_time=end,
        limit=None  # 不限制数量
    )


    # 处理数据
    processed_data = []
    # 用于聚合相同记录的字典
    aggregated_data = {}
    
    for event in result["data"]:
        # # 获取NOC工单信息
        # noc_case = event.get("noc_case")
        # # 确保noc_case不是None
        # if noc_case is None:
        #     noc_case = {}
        # noc_case_id = noc_case.get("noc_case_id", "")
        # outsource_ids = noc_case.get("outsource_ids", "")
        
        # # 只有当NOC工单和外包工单都不为空时，才进行处理
        # if not noc_case_id or not outsource_ids:
        #     continue

        # 获取主机名
        hostname = event.get("hostname", "non-data")
        remote_hostname = event.get("remote_hostname", "non-data")

        # 提取事件相关信息
        event_id = event.get("event_id", "non-data")
        event_type = event.get("event_type", "non-data")
        event_name = event.get("event_name", "non-data")
        
        # 提取时间字段
        starts_at = event.get("starts_at")
        claimed_at = event.get("claimed_at")
        case_at = event.get("case_at")
        operation_at = event.get("operation_at")
        isolate_at = event.get("isolate_at")
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
        operation_at_str = format_datetime(operation_at)
        isolate_at_str = format_datetime(isolate_at)
        ends_at_str = format_datetime(ends_at)
            
        # 获取光模块信息
        optical_module = event.get("optical_module")
        if optical_module is None:
            optical_module = {}
        
        # 检查是否一端是TOR，另一端是AGG
        is_tor_agg_pair = (
            re.search(r'-ROCE_TOR-0\d', hostname) and re.search(r'-ROCE_AGG-0\d', remote_hostname)
        ) or (
            re.search(r'-ROCE_AGG-0\d', hostname) and re.search(r'-ROCE_TOR-0\d', remote_hostname)
        )
        
        if is_tor_agg_pair:
            # 确定哪一端是TOR，哪一端是AGG
            if re.search(r'-ROCE_TOR-0\d', hostname):
                tor_hostname, agg_hostname = hostname, remote_hostname
                tor_interface_name, agg_interface_name = event.get("interface_name", "non-data"), event.get("remote_interface_name", "non-data")
                tor_module_name, agg_module_name = optical_module.get("module_name", "non-data"), optical_module.get("remote_module_name", "non-data")
                tor_module_vendor, agg_module_vendor = optical_module.get("module_vendor", "non-data"), optical_module.get("remote_module_vendor", "non-data")
                tor_idc, agg_idc = event.get("device_idc", "non-data"), event.get("remote_device_idc", "non-data")
            else:
                tor_hostname, agg_hostname = remote_hostname, hostname
                tor_interface_name, agg_interface_name = event.get("remote_interface_name", "non-data"), event.get("interface_name", "non-data")
                tor_module_name, agg_module_name = optical_module.get("remote_module_name", "non-data"), optical_module.get("module_name", "non-data")
                tor_module_vendor, agg_module_vendor = optical_module.get("remote_module_vendor", "non-data"), optical_module.get("module_vendor", "non-data")
                tor_idc, agg_idc = event.get("remote_device_idc", "non-data"), event.get("device_idc", "non-data")
            
            # 获取集群信息和客户信息
            tor_cluster_name, tor_customer_name = get_cluster_info(tor_hostname, is_tor=True)
            agg_cluster_name, _ = get_cluster_info(agg_hostname, is_agg=True)
            # ROCE-AGG下联的集群所属的客户和它对应的ROCE-TOR上联的客户一样
            agg_customer_name = tor_customer_name
            
            # 确保所有字段都有值，如果没有则替换为"non-data"
            tor_module_name = tor_module_name if tor_module_name else "non-data"
            tor_module_vendor = tor_module_vendor if tor_module_vendor else "non-data"
            tor_idc = tor_idc if tor_idc else "non-data"
            tor_interface_name = tor_interface_name if tor_interface_name else "non-data"
            tor_cluster_name = tor_cluster_name if tor_cluster_name else "non-data"
            tor_customer_name = tor_customer_name if tor_customer_name else "non-data"
            
            agg_module_name = agg_module_name if agg_module_name else "non-data"
            agg_module_vendor = agg_module_vendor if agg_module_vendor else "non-data"
            agg_idc = agg_idc if agg_idc else "non-data"
            agg_interface_name = agg_interface_name if agg_interface_name else "non-data"
            agg_cluster_name = agg_cluster_name if agg_cluster_name else "non-data"
            agg_customer_name = agg_customer_name if agg_customer_name else "non-data"
            
            # 为TOR端设备创建记录
            key = (event_id, event_type, event_name, starts_at_str, isolate_at_str, ends_at_str, tor_hostname, tor_interface_name, tor_module_name, tor_module_vendor, tor_idc, tor_cluster_name, tor_customer_name, claimed_at_str, case_at_str, operation_at_str)
            if key in aggregated_data:
                aggregated_data[key] += 1
            else:
                aggregated_data[key] = 1
            
            # 为AGG端设备创建记录
            key = (event_id, event_type, event_name, starts_at_str, isolate_at_str, ends_at_str, agg_hostname, agg_interface_name, agg_module_name, agg_module_vendor, agg_idc, agg_cluster_name, agg_customer_name, claimed_at_str, case_at_str, operation_at_str)
            if key in aggregated_data:
                aggregated_data[key] += 1
            else:
                aggregated_data[key] = 1
        else:
            # 对于非TOR-AGG对的设备，记录基本信息，集群和客户为空
            interface_name = event.get("interface_name", "non-data")
            module_name = optical_module.get("module_name", "non-data")
            module_vendor = optical_module.get("module_vendor", "non-data")
            idc = event.get("device_idc", "non-data")
            
            # 确保所有字段都有值，如果没有则替换为"non-data"
            interface_name = interface_name if interface_name else "non-data"
            module_name = module_name if module_name else "non-data"
            module_vendor = module_vendor if module_vendor else "non-data"
            idc = idc if idc else "non-data"
            
            # 为设备创建记录
            key = (event_id, event_type, event_name, starts_at_str, isolate_at_str, ends_at_str, hostname, interface_name, module_name, module_vendor, idc, "", "", claimed_at_str, case_at_str, operation_at_str)
            if key in aggregated_data:
                aggregated_data[key] += 1
            else:
                aggregated_data[key] = 1
    
    # 转换聚合数据为列表
    for (event_id, event_type, event_name, starts_at, isolate_at, ends_at, hostname, interface_name, module_name, module_vendor, idc, cluster, customer, claimed_at, case_at, operation_at), count in aggregated_data.items():
        processed_data.append({
            "事件ID": event_id,
            "事件类型": event_type,
            "事件名称": event_name,
            "开始时间": starts_at,
            "隔离时间": isolate_at,
            "结束时间": ends_at,
            "设备主机名": hostname,
            "设备端口名称": interface_name,
            "光模块型号": module_name,
            "光模块厂商": module_vendor,
            "机房": idc,
            "集群": cluster,
            "客户": customer,
            "故障数": count,
            "认领时间": claimed_at,
            "建单时间": case_at,
            "操作时间": operation_at
        })
    
    logger.info(f"处理完成，共生成 {len(processed_data)} 条记录")
    return processed_data


# ==========================================
# 主函数
# ==========================================
def main():
    try:
        # # 1. 旧基础网络事件列表查询
        # query_basicnet_event_demo()

        # 2. 事件监控查询
        query_event_monitor_demo()

        # # 3. ROCE事件查询
        # query_roce_event_demo()

    except Exception as e:
        logger.error(f"执行查询时出错: {e}")
        print(f"错误: {e}")

if __name__ == "__main__":
    main()
