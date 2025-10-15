#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件查询
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging
from mongodb_connector import connect_mongodb, get_database

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==========================================
# 基础网络事件查询
# ==========================================
def query_basicnet_event_demo():
    # 数据库连接配置
    basicnet_event_mongo_host = "10.69.74.235"
    basicnet_event_mongo_port  = 8526
    basicnet_event_mongo_db    = "alertpolicy"
    basicnet_event_mongo_user  = "network_event_ro_test"
    basicnet_event_mongo_pass  = "Ro123456"

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
    event_monitor_mongo_user  = "network_event_ro_test"
    event_monitor_mongo_pass  = "Ro123456"

    # 连接数据库
    if not connect_mongodb(host=event_monitor_mongo_host, port=event_monitor_mongo_port,
                          database=event_monitor_mongo_db,
                          username=event_monitor_mongo_user, password=event_monitor_mongo_pass):
        logger.error("事件监控数据库连接失败")
        return

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

    # 最近7天
    end = datetime.now()
    start = (end - timedelta(days=20)).replace(hour=0, minute=0, second=0, microsecond=0)
    print("查询时间范围:", start, "至", end)

    event_type = "NetworkDeviceInterface"
    is_finished = True  # True表示只查询已结束的事件, False表示只查询未结束的事件
    is_one_click_finish = False  # None表示不区分是否一键完成的事件, True表示只查询一键完成的事件, False表示只查询非一键完成的事件

    result = query_event_monitor_event_list(
        event_type=event_type,
        is_finished=is_finished,
        is_one_click_finish=is_one_click_finish,
        start_time=start,
        end_time=end,
        limit=None  # 不限制数量
    )

    print(result["data"])

    for event in result["data"]:
        print("事件ID:", event.get("event_id"),
              "事件类型:", event.get("event_type"),
              "开始时间:", event.get("starts_at"))
        print("光模块信息:", event.get("optical_module"))
        print()

    print("总记录数:", result["total"])

# ==========================================
# ROCE事件查询
# ==========================================
def query_roce_event_demo():
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
        return

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

    #最近7天
    end = datetime.now()
    start = (end - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
    print("查询时间范围:", start, "至", end)

    event_code = "CRC_ERROR"
    is_finished = None  # True表示只查询已结束的事件, False表示只查询未结束的事件
    is_one_click_finish = False  # None表示不区分是否一键完成的事件, True表示只查询一键完成的事件, False表示只查询非一键完成的事件

    result = query_roce_event_list(
        event_code=event_code,
        is_finished=is_finished,
        is_one_click_finish=is_one_click_finish,
        start_time=start,
        end_time=end,
        limit=None  # 不限制数量
    )

    # 按照原文件中的遍历方式
    for event in result["data"]:
        print("事件ID:", event.get("event_id"),
              "事件类型:", event.get("event_name"),
              "开始时间:", event.get("create_time"))
        print("光模块信息:", event.get("optical_model_vendor"),
              event.get("optical_model_name"),
              event.get("optical_model_sn"))
        print()

    print("总记录数:", result["total"])

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