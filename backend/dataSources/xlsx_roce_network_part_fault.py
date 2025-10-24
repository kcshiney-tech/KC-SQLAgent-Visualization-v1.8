#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理XLSX文件中的ROCE下联网络零件故障数据
"""

import pandas as pd
import logging
from typing import List, Dict, Any
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_xlsx_roce_network_part_fault(file_path: str) -> List[Dict[str, Any]]:
    """
    处理XLSX文件中的ROCE下联网络零件故障数据
    
    Args:
        file_path (str): XLSX文件路径
        
    Returns:
        List[Dict[str, Any]]: 处理后的数据列表
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return []
        
        # 读取XLSX文件
        logger.info(f"开始读取XLSX文件: {file_path}")
        df = pd.read_excel(file_path, header=0)
        
        # 检查必要的列是否存在
        required_columns = [
            "事件ID", "事件名称", "产生时间", "交换机名称", "交换机端口", "机房",
            "光模块厂商", "光模块型号", "光模块SN", "客户信息", "外包单号", "事件完成时间", "事件描述", "光模块是否更换"
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"缺少必要的列: {missing_columns}")
            return []
        
        # 筛选出"光模块是否更换"为"是"的记录
        filtered_df = df[df["光模块是否更换"] == "是"]
        logger.info(f"原始数据 {len(df)} 条记录，筛选后 {len(filtered_df)} 条记录")
        
        if filtered_df.empty:
            logger.warning("没有符合条件的记录")
            return []
        
        # 处理数据
        processed_data = []
        
        # 收集需要查询的SN
        sn_list = []
        sn_to_indices = {}  # 用于存储SN到记录索引的映射
        
        # 导入集群信息提取函数
        from backend.dataSources.cluster_info import get_cluster_info_from_hostname
        
        for index, row in filtered_df.iterrows():
            # 提取字段
            event_id = row.get("事件ID", "non-data")
            event_name = row.get("事件名称", "non-data")
            create_time = row.get("产生时间", "non-data")
            switch_name = row.get("交换机名称", "non-data")
            switch_port = row.get("交换机端口", "non-data")
            idc = row.get("机房", "non-data")
            part_producer = row.get("光模块厂商", "non-data")
            part_model = row.get("光模块型号", "non-data")
            part_sn = row.get("光模块SN", "non-data")
            customer_info = row.get("客户信息", "non-data")
            outsource_id = row.get("外包单号", "non-data")
            finish_time = row.get("事件完成时间", "non-data")
            event_desc = row.get("事件描述", "non-data")
            
            # 获取集群信息（使用ROCE事件模式）
            cluster, _ = get_cluster_info_from_hostname(switch_name, is_roce_event=True)
            
            # 确定网络零件种类（使用与roce_event中相同的逻辑）
            part_type = "光模块"  # 默认为光模块
            
            # 确保所有字段都是字符串类型
            part_model_str = str(part_model)
            part_producer_str = str(part_producer)
            part_sn_str = str(part_sn)
            
            # 如果网络零件型号中包含AOC字符串，或者网络零件厂商为TRILIGHT或包含"钧恒"且网络零件SN包含"-"，则标记为"AOC"
            if "AOC" in part_model_str.upper() or \
               part_producer_str.upper() == "TRILIGHT" or \
               ("钧恒" in part_producer_str and "-" in part_sn_str):
                part_type = "AOC"
            else:
                # 收集需要进一步查询的SN
                sn_list.append(part_sn_str)
            # 记录SN对应的记录索引
            if part_sn_str not in sn_to_indices:
                sn_to_indices[part_sn_str] = []
            sn_to_indices[part_sn_str].append(len(processed_data))
            
            # 创建记录
            record = {
                "事件ID": event_id,
                "事件名称": event_name,
                "产生时间": create_time,
                "交换机名称": switch_name,
                "交换机端口名称": switch_port,
                "机房": idc,
                "网络零件种类": part_type,
                "网络零件厂商": part_producer,
                "网络零件型号": part_model,
                "网络零件SN": part_sn,
                "集群": cluster,
                "客户信息": customer_info,
                "外包单号": outsource_id,
                "事件完成时间": finish_time,
                "事件描述": event_desc
            }
            
            processed_data.append(record)
        
        # 对于收集到的SN，使用API查询确定是否为AOC
        if sn_list:
            # 确保所有SN都是字符串类型
            str_sn_list = [str(sn) for sn in sn_list]
            
            # 去重
            unique_sns = sorted(list(set(str_sn_list)))  # 排序以提高效率
            logger.info(f"需要查询 {len(unique_sns)} 个唯一的SN")
            
            if unique_sns:
                # 导入API查询函数
                from backend.dataSources.noc_asset_query import query_cable_assets
                
                # 构造批量查询字符串
                batch_query = "\n".join(unique_sns)
                
                # 查询电缆资产
                cable_result = query_cable_assets(batch_query)
                
                # 如果查询成功，更新相应的记录
                if "error" not in cable_result:
                    cable_data = cable_result.get("data", [])
                    logger.info(f"API查询返回 {len(cable_data)} 条记录")
                    
                    # 更新标记为AOC的记录
                    for cable_record in cable_data:
                        sn_a = str(cable_record.get("sn_a", ""))  # 确保是字符串类型
                        sn_b = str(cable_record.get("sn_b", ""))  # 确保是字符串类型
                        
                        # 检查A端SN
                        if sn_a in sn_to_indices:
                            for idx in sn_to_indices[sn_a]:
                                processed_data[idx]["网络零件种类"] = "AOC"
                        
                        # 检查B端SN
                        if sn_b in sn_to_indices:
                            for idx in sn_to_indices[sn_b]:
                                processed_data[idx]["网络零件种类"] = "AOC"
                
        logger.info(f"数据处理完成，共生成 {len(processed_data)} 条记录")
        return processed_data
        
    except Exception as e:
        logger.error(f"处理XLSX文件时出错: {e}")
        raise

def main():
    """主函数"""
    # 示例用法
    file_path = "示例文件.xlsx"  # 实际使用时请替换为真实的文件路径
    
    try:
        data = process_xlsx_roce_network_part_fault(file_path)
        
        if data:
            print(f"成功处理 {len(data)} 条记录")
            print("前5条记录:")
            for i, record in enumerate(data[:5]):
                print(f"  {i+1}. {record}")
        else:
            print("未能处理数据")
            
    except Exception as e:
        logger.error(f"执行时出错: {e}")
        print(f"错误: {e}")

if __name__ == "__main__":
    main()
