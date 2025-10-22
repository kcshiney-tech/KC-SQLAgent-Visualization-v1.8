#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOC光模块全量数据拉取工具
支持并发拉取和处理光模块数据，与optical_module_inventory.py中的逻辑一致
"""

import sys
import os
import logging
from typing import Dict, Any, List
from datetime import datetime
from collections import defaultdict

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def fetch_and_process_optical_modules_full() -> List[Dict[str, Any]]:
    """
    拉取并处理NOC中光模块全量数据，构建与optical_module_inventory.py中相同的表结构
    
    Returns:
        List[Dict[str, Any]]: 处理后的光模块库存数据
    """
    try:
        # 导入noc_asset_query模块
        from backend.dataSources.noc_asset_query import fetch_optical_modules_full
        from backend.dataSources.cluster_info import get_cluster_info_from_hostname
        
        # 拉取全量数据
        logger.info("开始拉取NOC光模块全量数据...")
        raw_data = fetch_optical_modules_full(start_id=0, page_length=1000)
        logger.info(f"成功拉取 {len(raw_data)} 条原始数据")
        
        # 处理数据
        processed_data = []
        # 用于聚合相同记录的字典
        aggregated_data = {}
        
        # 获取当前时间作为日期字段
        current_time = datetime.now().isoformat()
        
        for item in raw_data:
            # 提取字段
            idc = item.get("idc", "non-data")
            module_type = item.get("type", "non-data")
            producer = item.get("producer", "non-data")
            hostname = item.get("device", "non-data")
            status = item.get("status", "non-data")
            
            # 处理光模块型号
            processed_module_type = process_module_type(module_type)
            
            # 获取集群和客户信息
            cluster, customer = get_cluster_info_from_hostname(hostname, is_roce_event=False)
            
            # 创建聚合键
            key = (processed_module_type, producer, status, idc, cluster, customer)
            if key in aggregated_data:
                aggregated_data[key] += 1
            else:
                aggregated_data[key] = 1
        
        # 转换聚合数据为列表
        for (module_type, producer, status, idc, cluster, customer), count in aggregated_data.items():
            processed_data.append({
                "日期": current_time,
                "光模块型号": module_type,
                "光模块厂商": producer,
                "状态" : status,
                # "主机名": hostname,
                "机房": idc,
                "集群": cluster,
                "客户": customer,
                "在线总数量": count
            })
        
        logger.info(f"数据处理完成，共生成 {len(processed_data)} 条记录")
        return processed_data
        
    except Exception as e:
        logger.error(f"拉取和处理光模块数据时出错: {e}")
        raise

def save_to_csv(data: List[Dict[str, Any]], filename: str = "optical_modules_full_output.csv") -> bool:
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
    print("NOC光模块全量数据拉取工具")
    print("=" * 40)
    
    try:
        # 拉取并处理数据
        data = fetch_and_process_optical_modules_full()
        
        if data:
            print(f"成功获取 {len(data)} 条记录")
            print("前5条记录:")
            for i, record in enumerate(data[:5]):
                print(f"  {i+1}. {record}")
            
            # 保存到CSV文件
            save_success = save_to_csv(data, "optical_modules_full_output.csv")
            if save_success:
                print("\n数据已保存到 optical_modules_full_output.csv")
            else:
                print("\n数据保存失败")
        else:
            print("未能获取数据")
            
    except Exception as e:
        logger.error(f"执行时出错: {e}")
        print(f"错误: {e}")

if __name__ == "__main__":
    main()
