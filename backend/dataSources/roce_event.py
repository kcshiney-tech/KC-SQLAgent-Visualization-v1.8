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
from backend.dataSources.noc_asset_query import query_cable_assets
from backend.dataSources.cluster_info import get_cluster_info_from_hostname

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==========================================
# ROCE事件查询
# ==========================================
def query_roce_network_event_demo():
    # 数据库连接配置
    roce_event_mongo_host = "10.69.74.235"
    roce_event_mongo_port  = 8526
    roce_event_mongo_db    = "alertpolicy"
    roce_event_mongo_user  = "network_event_ro_user_sysnet"
    roce_event_mongo_pass  = "rosysnet5678"

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

    #最近360天
    days = 360
    end = datetime.now()
    start = (end - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    logger.info(f"查询时间范围: {start} 至 {end}")

    event_code = None
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
    
    # 收集需要查询的SN
    sn_list = []
    sn_to_events = {}  # 用于存储SN到事件索引的映射
    
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
        
        # # 只有光模块是否更换为true的才被计入数据库中
        # if model_changed != "是":
        #     continue
        
        # 获取集群信息
        cluster_name, _ = get_cluster_info_from_hostname(hostname, is_roce_event=True)
        
        # 初始化网络零件种类为光模块
        part_type = "光模块"
        
        # 判断网络零件种类
        # 如果网络零件型号中包含AOC字符串，或者网络零件厂商为TRILIGHT或包含"钧恒"且网络零件SN包含"-"，则标记为"AOC"
        if "AOC" in str(optical_model_name).upper() or \
           str(optical_model_vendor).upper() == "TRILIGHT" or \
           ("钧恒" in str(optical_model_vendor) and "-" in str(optical_model_sn)):
            part_type = "AOC"
        else:
            # 收集需要进一步查询的SN
            sn_list.append(optical_model_sn)
            # 记录SN对应的事件索引
            if optical_model_sn not in sn_to_events:
                sn_to_events[optical_model_sn] = []
            sn_to_events[optical_model_sn].append(len(processed_data))
        
        # 创建记录
        processed_data.append({
            "事件ID": event_id,
            "事件名称": event_name,
            "产生时间": create_time,
            "交换机名称": hostname,
            "交换机端口名称": portname,
            "机房": idc,
            "网络零件种类": part_type,
            "网络零件厂商": optical_model_vendor,
            "网络零件型号": optical_model_name,
            "网络零件SN": optical_model_sn,
            "集群": cluster_name,
            "客户信息": customer_name,
            "外包单号": last_outsource_ids,
            "事件完成时间": end_time
        })
    
    # 对于收集到的SN，使用API查询确定是否为AOC
    if sn_list:
        # 去重
        unique_sns = sorted(list(set(sn_list)))  # 排序以提高效率
        # 构造批量查询字符串
        batch_query = "\n".join(unique_sns)
        # 查询电缆资产
        cable_result = query_cable_assets(batch_query)
        
        # 如果查询成功，更新相应的记录
        if "error" not in cable_result:
            cable_data = cable_result.get("data", [])
            # 按SN排序以提高查找效率
            sorted_cable_data = sorted(cable_data, key=lambda x: (x.get("sn_a", ""), x.get("sn_b", "")))
            
            # 更新标记为AOC的记录
            for cable_record in sorted_cable_data:
                sn_a = cable_record.get("sn_a", "")
                sn_b = cable_record.get("sn_b", "")
                
                # 检查A端SN
                if sn_a in sn_to_events:
                    for index in sn_to_events[sn_a]:
                        processed_data[index]["网络零件种类"] = "AOC"
                
                # 检查B端SN
                if sn_b in sn_to_events:
                    for index in sn_to_events[sn_b]:
                        processed_data[index]["网络零件种类"] = "AOC"
    
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
