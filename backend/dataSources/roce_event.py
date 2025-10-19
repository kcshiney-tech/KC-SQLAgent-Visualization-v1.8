#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROCE事件查询
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
from backend.dataSources.mongodb_connector import connect_mongodb, get_database
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_cluster_info(hostname: str) -> tuple:
    """
    根据主机名确定集群名和客户名称
    对于ROCE网络事件，只有ROCE-TOR下联-集群编号0x，如果主机名符合-ROCE_TOR-0x，不用检查对端，直接就是这个集群名。
    
    Args:
        hostname (str): 主机名
        
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
            
            # 对于ROCE网络事件，集群名格式为ROCE-TOR下联-集群编号0x
            cluster_name = f"ROCE-TOR下联-{cluster_name}"
            return (cluster_name, customer_name)
    
    # 对于其他主机名，检查是否符合-ROCE_TOR-0x模式
    tor_match = re.match(r'^([A-Z]+)-([A-Z0-9]+)-[A-Z0-9-]*ROCE_TOR-(\d+)', hostname)
    
    if tor_match:
        part1, part2, number = tor_match.groups()
        cluster_name = f"{part1}{part2}0{number[-1]}"
        # 处理重复字符的情况
        if len(cluster_name) >= 6 and cluster_name[:2] == cluster_name[2:4]:
            cluster_name = cluster_name[2:]
        # 对于ROCE网络事件，集群名格式为ROCE-TOR下联-集群编号0x
        cluster_name = f"ROCE-TOR下联-{cluster_name}"
        return (cluster_name, "no-data")
    
    # 没匹配上集群名的，集群和客户处为空
    return ("", "")


# ==========================================
# ROCE事件查询
# ==========================================
def query_roce_network_event_demo():
    # 数据库连接配置
    roce_event_mongo_host = "10.69.74.235"
    roce_event_mongo_port  = 8526
    roce_event_mongo_db    = "alertpolicy"
    roce_event_mongo_user  = "network_event_ro_test"
    roce_event_mongo_pass  = "Ro123456"

    # 连接数据库
    if not connect_mongodb(host=roce_event_mongo_host, port=roce_event_mongo_port,
                          database=roce_event_mongo_db,
                          username=roce_event_mongo_user, password=roce_event_mongo_pass):
        logger.error("ROCE事件数据库连接失败")
        return []

    # 通用查询方法（简化版）
    def query_roce_event_list(
        event_code: Optional[str] = None,
        is_finished: Optional[bool] = None,
        is_one_click_finish: Optional[bool] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        database: str = "alertpolicy",
        collection: str = "roce_event"
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
                query["event_code"] = {"$nin": ["GPU_ABNORMAL", "SWITCH_DOWN"]}

            if is_finished is not None:
                if is_finished:
                    query["process_code"] = 40
                else:
                    query["process_code"] = {"$lt": 40}

            if is_one_click_finish is not None:
                if is_one_click_finish:
                    query["quick_finish"] = "是"
                else:
                    query["quick_finish"] = {"$ne": "是"}

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

    #最近90天
    end = datetime.now()
    start = (end - timedelta(days=90)).replace(hour=0, minute=0, second=0, microsecond=0)
    logger.info(f"查询时间范围: {start} 至 {end}")

    event_code = "CRC_ERROR"
    is_finished = True  # None表示不区分是否结束的事件
    is_one_click_finish = None  # None表示不区分是否一键完成的事件

    result = query_roce_event_list(
        event_code=event_code,
        is_finished=is_finished,
        is_one_click_finish=is_one_click_finish,
        start_time=start,
        end_time=end,
        limit=None  # 不限制数量
    )

    # 处理数据
    processed_data = []
    
    for event in result["data"]:
        # 提取事件相关信息
        event_id = event.get("event_id", "non-data")
        event_name = event.get("event_name", "non-data")
        create_time = event.get("create_time", "non-data")
        hostname = event.get("hostname", "non-data")
        portname = event.get("portname", "non-data")
        idc = event.get("idc", "non-data")
        optical_model_vendor = event.get("optical_model_vendor", "non-data")
        optical_model_name = event.get("optical_model_name", "non-data")
        optical_model_sn = event.get("optical_model_sn", "non-data")
        model_changed = event.get("model_changed", "non-data")
        last_outsource_ids = event.get("last_outsource_ids", "non-data")
        end_time = event.get("end_time", "non-data")
        # 直接从数据库字段获取客户信息
        customer_name = event.get("server_user", "non-data")
        
        # 只有光模块是否更换为true的才被计入数据库中
        if model_changed != "是":
            continue
        
        # 获取集群信息
        cluster_name, _ = get_cluster_info(hostname)
        
        # 创建记录
        processed_data.append({
            "事件ID": event_id,
            "事件名称": event_name,
            "产生时间": create_time,
            "交换机名称": hostname,
            "交换机端口名称": portname,
            "机房": idc,
            "光模块厂商": optical_model_vendor,
            "光模块型号": optical_model_name,
            "光模块SN": optical_model_sn,
            "光模块是否更换": model_changed,
            "集群": cluster_name,
            "客户信息": customer_name,
            "外包单号": last_outsource_ids,
            "事件完成时间": end_time,
            "事件描述": ""  # 暂时为空，可以根据需要填充
        })
    
    logger.info(f"处理完成，共生成 {len(processed_data)} 条记录")
    return processed_data


def main():
    try:
        # ROCE事件查询
        data = query_roce_network_event_demo()
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
