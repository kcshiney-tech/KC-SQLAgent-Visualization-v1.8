#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集群信息提取模块
用于从主机名中提取集群信息的共享模块
"""

import re
from typing import Tuple

# 已知的集群映射
CLUSTER_MAPPING = {
    "QH-QHDX-AZ-ROCE_TOR-02": ("QHDX02", "小米"),
    "NB-LT-AZ-ROCE_TOR-01": ("NBLT01", "百川"),
    "QY-YD-DC-ROCE_TOR-05": ("QYYD05", "月暗"),
    "QY-ZNJ-DC-ROCE_TOR-01": ("QYZNJ01", "云启")
}

def _process_idc_string(part1: str, part2: str) -> str:
    """
    处理机房字符串，如果part1和part2从第一个字符开始有重叠，则只取part2
    否则拼接part1和part2
    
    Args:
        part1 (str): 第一个字符串
        part2 (str): 第二个字符串
        
    Returns:
        str: 处理后的机房字符串
    """
    # 检查part2是否以part1开头（从第一个字符开始完全重叠）
    if part2.startswith(part1):
        # 只取part2
        return part2
    else:
        # 拼接part1和part2
        return part1 + part2

def get_cluster_info_from_hostname(hostname: str, is_roce_event: bool = False) -> Tuple[str, str]:
    """
    根据主机名确定集群名和客户名称
    
    Args:
        hostname (str): 主机名
        is_roce_event (bool): 是否为ROCE事件，如果是则只需看当前主机名
        
    Returns:
        tuple: (cluster_name, customer_name)
    """
    if not hostname or hostname == "non-data":
        return ("", "")
    
    # 检查是否匹配已知的集群
    for prefix, (cluster_name, customer_name) in CLUSTER_MAPPING.items():
        if hostname.startswith(prefix):
            # 处理重复字符的情况，如QHQHYD0x -> QHYD0x
            if len(cluster_name) >= 6 and cluster_name[:2] == cluster_name[2:4]:
                cluster_name = cluster_name[2:]
            
            # 根据是否为ROCE事件和主机名中的关键字添加前缀
            if is_roce_event:
                # 对于ROCE网络事件，只需看当前主机名
                if "-SROCE_TOR-" in hostname:
                    cluster_name = f"SROCE-TOR下联-{cluster_name}"
                elif "-ROCE_TOR-" in hostname:
                    cluster_name = f"ROCE-TOR下联-{cluster_name}"
            else:
                # 对于其他情况，根据主机名添加前缀
                if "-SROCE_TOR-" in hostname:
                    cluster_name = f"SROCE-TOR-{cluster_name}"
                elif "-ROCE_TOR-" in hostname:
                    cluster_name = f"ROCE-TOR-{cluster_name}"
                elif "-SROCE_AGG-" in hostname:
                    cluster_name = f"SROCE-AGG-{cluster_name}"
                elif "-ROCE_AGG-" in hostname:
                    cluster_name = f"ROCE-AGG-{cluster_name}"
                elif "-SROCE_CORE-" in hostname:
                    cluster_name = f"SROCE-CORE-{cluster_name}"
                elif "-ROCE_CORE-" in hostname:
                    cluster_name = f"ROCE-CORE-{cluster_name}"
                
            return (cluster_name, customer_name)
    
    # 对于ROCE事件，只需检查当前主机名
    if is_roce_event:
        # 检查是否匹配SROCE_TOR模式 (格式如 QY-YD-DC-SROCE_TOR-05_01_17_05.QY)
        stor_match = re.match(r'^([A-Z]+)-([A-Z0-9]+)-[A-Z0-9-]*SROCE_TOR-(\d+)_(\d+)_(\d+)_(\d+)(\.[A-Z0-9]+)?$', hostname)
        if stor_match:
            part1, part2, num, _, _, _, _ = stor_match.groups()
            # 处理机房字符串
            idc = _process_idc_string(part1, part2)
            cluster_name = f"SROCE-TOR下联-{idc}{num}"
            return (cluster_name, "")
            
        # 检查是否匹配ROCE_TOR模式 (格式如 QY-YD-DC-ROCE_TOR-05_01_17_05.QY)
        tor_match = re.match(r'^([A-Z]+)-([A-Z0-9]+)-[A-Z0-9-]*ROCE_TOR-(\d+)_(\d+)_(\d+)_(\d+)(\.[A-Z0-9]+)?$', hostname)
        if tor_match:
            part1, part2, num, _, _, _, _ = tor_match.groups()
            # 处理机房字符串
            idc = _process_idc_string(part1, part2)
            cluster_name = f"ROCE-TOR下联-{idc}{num}"
            return (cluster_name, "")
    else:
        # 对于其他情况，检查各种模式
        # 检查是否匹配SROCE_TOR模式
        stor_match = re.match(r'^([A-Z]+)-([A-Z0-9]+)-[A-Z0-9-]*SROCE_TOR-(\d+)_(\d+)_(\d+)_(\d+)(\.[A-Z0-9]+)?$', hostname)
        if stor_match:
            part1, part2, num, _, _, _, _ = stor_match.groups()
            # 处理机房字符串
            idc = _process_idc_string(part1, part2)
            cluster_name = f"SROCE-TOR-{idc}{num}"
            return (cluster_name, "")
            
        # 检查是否匹配ROCE_TOR模式 (更通用的格式)
        tor_match = re.match(r'^([A-Z]+)-([A-Z0-9]+)-[A-Z0-9-]*ROCE_TOR-(\d+)_(\d+)_(\d+)_(\d+)(\.[A-Z0-9]+)?$', hostname)
        if tor_match:
            part1, part2, num, _, _, _, _ = tor_match.groups()
            # 处理机房字符串
            idc = _process_idc_string(part1, part2)
            cluster_name = f"ROCE-TOR-{idc}{num}"
            return (cluster_name, "")
            
        # 检查是否匹配SROCE_AGG模式
        sagg_match = re.match(r'^([A-Z]+)-([A-Z0-9]+)-[A-Z0-9-]*SROCE_AGG-(\d+)_(\d+)_(\d+)_(\d+)(\.[A-Z0-9]+)?$', hostname)
        if sagg_match:
            part1, part2, num, _, _, _, _ = sagg_match.groups()
            # 处理机房字符串
            idc = _process_idc_string(part1, part2)
            cluster_name = f"SROCE-AGG-{idc}{num}"
            return (cluster_name, "")
            
        # 检查是否匹配ROCE_AGG模式
        agg_match = re.match(r'^([A-Z]+)-([A-Z0-9]+)-[A-Z0-9-]*ROCE_AGG-(\d+)_(\d+)_(\d+)_(\d+)(\.[A-Z0-9]+)?$', hostname)
        if agg_match:
            part1, part2, num, _, _, _, _ = agg_match.groups()
            # 处理机房字符串
            idc = _process_idc_string(part1, part2)
            cluster_name = f"ROCE-AGG-{idc}{num}"
            return (cluster_name, "")
            
        # 检查是否匹配SROCE_CORE模式
        score_match = re.match(r'^([A-Z]+)-([A-Z0-9]+)-[A-Z0-9-]*SROCE_CORE-(\d+)_(\d+)_(\d+)_(\d+)(\.[A-Z0-9]+)?$', hostname)
        if score_match:
            part1, part2, num, _, _, _, _ = score_match.groups()
            # 处理机房字符串
            idc = _process_idc_string(part1, part2)
            cluster_name = f"SROCE-CORE-{idc}{num}"
            return (cluster_name, "")
            
        # 检查是否匹配ROCE_CORE模式
        core_match = re.match(r'^([A-Z]+)-([A-Z0-9]+)-[A-Z0-9-]*ROCE_CORE-(\d+)_(\d+)_(\d+)_(\d+)(\.[A-Z0-9]+)?$', hostname)
        if core_match:
            part1, part2, num, _, _, _, _ = core_match.groups()
            # 处理机房字符串
            idc = _process_idc_string(part1, part2)
            cluster_name = f"ROCE-CORE-{idc}{num}"
            return (cluster_name, "")
    
    # 没匹配上集群名的，集群和客户处为空
    return ("", "")

def main():
    """主函数，用于测试"""
    # 测试用例
    test_cases = [
        "QY-YD-DC-ROCE_TOR-05_01_17_05.QY",  # 应该得到 ROCE-TOR下联-QYYD05
        "NX-LT-DC-ROCE_TOR-103_03.NX",      # 无法提取集群编号,集群为空
        "QH-QHDX-AZ-ROCE_TOR-02",           # 已知集群映射
        "NB-LT-AZ-ROCE_TOR-01",             # 已知集群映射
        "QY-YD-DC-ROCE_TOR-05",             # 已知集群映射
        "QY-ZNJ-DC-ROCE_TOR-01",            # 已知集群映射
        "QH-QHYD-TEST-ROCE_TOR-03",         # QH和QHYD重叠，应该只取QHYD
    ]
    
    print("测试集群信息提取:")
    for hostname in test_cases:
        cluster, customer = get_cluster_info_from_hostname(hostname, is_roce_event=True)
        print(f"主机名: {hostname}")
        print(f"  集群: {cluster}")
        print(f"  客户: {customer}")
        print()

if __name__ == "__main__":
    main()
