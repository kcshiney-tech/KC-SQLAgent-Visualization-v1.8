#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试集群信息提取模块
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataSources.cluster_info import get_cluster_info_from_hostname

def test_cluster_info():
    """测试集群信息提取"""
    test_cases = [
        "QY-YD-DC-SROCE_TOR-05_01_01_12.QY",  # 应该得到 SROCE-TOR下联-QYYD05
        "QY-YD-DC-ROCE_TOR-05_01_17_05.QY",   # 应该得到 ROCE-TOR下联-QYYD05
        "QY-YD-DC-SROCE_TOR-05_01_01_12",     # 应该得到 SROCE-TOR下联-QYYD05
        "QY-YD-DC-ROCE_TOR-05_01_17_05",      # 应该得到 ROCE-TOR下联-QYYD05
    ]
    
    print("测试集群信息提取:")
    for hostname in test_cases:
        cluster, customer = get_cluster_info_from_hostname(hostname, is_roce_event=True)
        print(f"主机名: {hostname}")
        print(f"  集群: {cluster}")
        print(f"  客户: {customer}")
        print()

if __name__ == "__main__":
    test_cluster_info()
