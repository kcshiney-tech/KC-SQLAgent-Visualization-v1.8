#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修改后的光学故障数据处理
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from backend.dataSources.optical_failure import get_cluster_info

def test_get_cluster_info():
    """测试集群信息获取函数"""
    print("测试集群信息获取函数...")
    
    # 测试已知集群
    cluster, customer = get_cluster_info("QH-QHDX-AZ-ROCE_TOR-02-xxx.QH", is_tor=True)
    print(f"QH-QHDX-AZ-ROCE_TOR-02-xxx.QH (TOR): 集群={cluster}, 客户={customer}")
    
