#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MongoDB连接模块"""

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局连接对象
_client: Optional[MongoClient] = None

def connect_mongodb(host: str = "localhost", port: int = 27017,
                   username: str = "", password: str = "",
                   database: str = "test") -> bool:
    """
    连接MongoDB数据库

    Args:
        host: MongoDB服务器地址
        port: MongoDB端口
        username: 用户名（可选）
        password: 密码（可选）
        database: 数据库名称

    Returns:
        bool: 连接成功返回True 失败返回False
    """
    global _client

    try:
        # 构建连接字符串
        if username and password:
            uri = f"mongodb://{username}:{password}@{host}:{port}/{database}"
        else:
            uri = f"mongodb://{host}:{port}/{database}"

        # 创建连接
        _client = MongoClient(uri, 
                              maxPoolSize=50,
                              minPoolSize=5,
                              serverSelectionTimeoutMS=5000)
        _client.admin.command('ping')

        logger.info(f"MongoDB连接成功: {host}:{port}/{database}")
        return True

    except ConnectionFailure as e:
        logger.error(f"MongoDB连接失败: {e}")
        return False
    except Exception as e:
        logger.error(f"连接错误: {e}")
        return False

def get_mongodb_connection():
    """
    获取MongoDB连接对象

    Returns:
        MongoClient: 连接对象 如果未连接则返回None
    """
    global _client
    if _client is None:
        logger.warning("MongoDB未连接 请先调用connect_mongodb()")
    return _client

def get_database(database: str = "test"):
    """
    获取数据库对象

    Args:
        database: 数据库名称

    Returns:
        Database: 数据库对象 如果未连接则返回None
    """
    global _client
    if _client is None:
        logger.warning("MongoDB未连接 请先调用connect_mongodb()")
        return None
    return _client[database]

def close_mongodb():
    """关闭MongoDB连接"""
    global _client
    if _client:
        _client.close()
        _client = None
        logger.info("MongoDB连接已关闭")

# 使用示例
if __name__ == "__main__":
    basicnet_mongo_host = "10.69.72.235"
    basicnet_mongo_port  = 8526
    basicnet_mongo_db    = "network_event"

    # 连接MongoDB（有认证）
    # connect_mongodb(host="localhost", port=27017,
    #                username="admin", password="password",
    #                database="mydb")

    if connect_mongodb(host=basicnet_mongo_host, port=basicnet_mongo_port, database=basicnet_mongo_db):
        # 获取数据库
        db = get_database(basicnet_mongo_db)

        # 使用数据库...
        print("MongoDB连接成功 可以使用数据库了")

        # 关闭连接
        close_mongodb()

