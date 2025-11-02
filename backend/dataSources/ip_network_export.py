#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IP网络信息导出到Excel工具
"""

import sys
import os
import pandas as pd
from typing import List, Dict
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.dataSources.noc_asset_query import fetch_ip_networks_full

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# IDC ID 映射表
IDC_ID_MAPPING = {
    "TJYXY01": 1867,
    # 可以添加更多机房映射
    # "BJYXY01": 1868,
    # "SHYXY01": 1869,
}

# 字段映射关系
FIELD_MAPPING = {
    "name": "网段",
    "mask": "子网掩码",
    "gateway": "网关",
    "parent": "所属网段",
    "belong_type_": "归属类型",
    "belong_": "地址归属",
    "type_": "内外网",
    "status_id_": "状态",
    "isp": "运营商",
    "charger": "接口人",
    "usage": "地址用途",
    "remark": "备注",
    "idc": "机房信息"
}

def export_ip_networks_to_excel(idc_id: int = 1867, output_file: str = "ip_networks.xlsx") -> bool:
    """
    导出IP网络信息到Excel文件
    
    Args:
        idc_id (int): 机房ID，默认为1867
        output_file (str): 输出文件名，默认为"ip_networks.xlsx"
        
    Returns:
        bool: 导出是否成功
    """
    try:
        # 调用API获取全量数据
        logger.info(f"开始获取IDC {idc_id} 的全量IP网络信息...")
        data = fetch_ip_networks_full(idc_id=idc_id)
        
        if not data:
            logger.warning("未获取到任何IP网络数据")
            # 创建空的DataFrame
            df = pd.DataFrame(columns=list(FIELD_MAPPING.values()))
        else:
            logger.info(f"成功获取 {len(data)} 条IP网络记录")
            
            # 转换数据格式
            processed_data = []
            for record in data:
                # 根据字段映射创建新记录
                new_record = {}
                for eng_field, ch_field in FIELD_MAPPING.items():
                    new_record[ch_field] = record.get(eng_field, "")
                processed_data.append(new_record)
            
            # 创建DataFrame
            df = pd.DataFrame(processed_data)
        
        # 导出到Excel
        logger.info(f"正在导出数据到 {output_file}...")
        df.to_excel(output_file, index=False, engine='openpyxl')
        logger.info(f"数据已成功导出到 {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"导出过程中发生错误: {e}")
        return False

def main():
    """主函数"""
    print("IP网络信息导出工具")
    print("=" * 30)
    
    # 默认参数
    idc_id = 1867
    output_file = "ip_networks.xlsx"
    
    # 显示可用的IDC映射
    print("可用的IDC映射:")
    for idc_name, idc_id_val in IDC_ID_MAPPING.items():
        print(f"  {idc_name}: {idc_id_val}")
    print()
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        # 尝试将参数解析为IDC名称
        if sys.argv[1] in IDC_ID_MAPPING:
            idc_id = IDC_ID_MAPPING[sys.argv[1]]
            print(f"使用IDC名称 '{sys.argv[1]}' 对应的ID: {idc_id}")
        else:
            # 尝试解析为数字ID
            try:
                idc_id = int(sys.argv[1])
                print(f"使用指定的IDC ID: {idc_id}")
            except ValueError:
                print(f"无效的IDC参数: {sys.argv[1]}，使用默认值 {idc_id}")
    
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print(f"\n参数设置:")
    print(f"  IDC ID: {idc_id}")
    print(f"  输出文件: {output_file}")
    print("=" * 30)
    
    # 执行导出
    success = export_ip_networks_to_excel(idc_id, output_file)
    
    if success:
        print("导出成功!")
    else:
        print("导出失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()
