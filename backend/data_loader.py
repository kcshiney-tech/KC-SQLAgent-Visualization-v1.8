# data_loader.py
import pandas as pd
import requests
from abc import ABC, abstractmethod
import logging
from typing import List, Dict, Any, Tuple, Optional
import os
import traceback  # 添加以追踪错误

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_loader.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

class DataSourceLoader(ABC):
    """Abstract base class for loading data from various sources."""
    
    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load data and return a list of dictionaries with table names and records.
        
        Returns:
            List[Dict[str, Any]]: List of dicts with 'table_name' and 'data' keys.
        Raises:
            Exception: If data loading fails (specific exceptions in implementations).
        """
        pass

class ExcelDataSourceLoader(DataSourceLoader):
    """Loader for Excel (.xlsx) files, supporting specific sheets and custom table names."""
    
    def __init__(self, file_path: str, sheets: Optional[List[Tuple[str, Optional[str]]]] = None):
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path} (status=1)")
                raise FileNotFoundError(f"Excel file not found: {file_path}")
            if sheets is not None:
                if not all(isinstance(s, tuple) and len(s) == 2 for s in sheets):
                    logger.error("Invalid sheets format: must be list of (sheet_name, table_name) tuples (status=1)")
                    raise ValueError("sheets must be a list of (sheet_name, table_name) tuples")
                if not all(s[0].strip() for s in sheets):
                    logger.error("Sheet names cannot be empty (status=1)")
                    raise ValueError("Sheet names cannot be empty")
                if not all(s[1].strip() if s[1] is not None else True for s in sheets):
                    logger.error("Table names cannot be empty when specified (status=1)")
                    raise ValueError("Table names cannot be empty when specified")
            self.file_path = file_path
            self.sheets = sheets
            logger.debug(f"Initialized ExcelDataSourceLoader for {file_path} with sheets: {sheets}")
        except Exception as e:
            logger.error(f"Initialization failed: {traceback.format_exc()}")
            raise

    def load_data(self) -> List[Dict[str, Any]]:
        try:
            logger.debug(f"Loading Excel file: {self.file_path}")
            xl = pd.ExcelFile(self.file_path)
            sheet_configs = self.sheets if self.sheets is not None else [(name, None) for name in xl.sheet_names]
            data = []
            for sheet_name, table_name in sheet_configs:
                if sheet_name not in xl.sheet_names:
                    logger.warning(f"Sheet {sheet_name} not found in {self.file_path} (status=1)")
                    continue
                logger.debug(f"Reading sheet: {sheet_name}")
                df = pd.read_excel(self.file_path, sheet_name=sheet_name, header=0)
                if df.empty:
                    logger.warning(f"Sheet {sheet_name} in {self.file_path} is empty (status=1)")
                    continue
                columns = df.columns.tolist()
                logger.debug(f"Columns in sheet {sheet_name}: {columns}")
                if any(not str(col).strip() for col in columns):
                    logger.error(f"Invalid (empty) column names in sheet {sheet_name}: {columns} (status=1)")
                    raise ValueError(f"Sheet {sheet_name} contains empty column names")
                effective_table_name = table_name if table_name is not None else sheet_name
                data.append({"table_name": effective_table_name, "data": df.to_dict(orient="records")})
                logger.debug(f"Loaded sheet {sheet_name} as table {effective_table_name} with {len(df)} records")
            logger.info(f"Loaded Excel file {self.file_path} with sheets: {[s[0] for s in sheet_configs]} (status=0)")
            return data
        except Exception as e:
            logger.error(f"Failed to load Excel file {self.file_path}: {traceback.format_exc()} (status=1)")
            raise
        finally:
            logger.info(f"Excel load completed for {self.file_path} (status=0)")

class CSVDataSourceLoader(DataSourceLoader):  # 新添加：支持CSV上传
    """Loader for CSV files, treating the file as a single table."""
    
    def __init__(self, file_path: str, table_name: str):
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path} (status=1)")
                raise FileNotFoundError(f"CSV file not found: {file_path}")
            if not table_name.strip():
                logger.error("Table name cannot be empty (status=1)")
                raise ValueError("Table name cannot be empty")
            self.file_path = file_path
            self.table_name = table_name
            logger.debug(f"Initialized CSVDataSourceLoader for {file_path} with table {table_name}")
        except Exception as e:
            logger.error(f"Initialization failed: {traceback.format_exc()}")
            raise

    def load_data(self) -> List[Dict[str, Any]]:
        try:
            logger.debug(f"Loading CSV file: {self.file_path}")
            df = pd.read_csv(self.file_path, header=0)
            if df.empty:
                logger.warning(f"CSV {self.file_path} is empty (status=1)")
                return []
            columns = df.columns.tolist()
            logger.debug(f"Columns in CSV: {columns}")
            if any(not str(col).strip() for col in columns):
                logger.error(f"Invalid (empty) column names in CSV: {columns} (status=1)")
                raise ValueError("CSV contains empty column names")
            data = [{"table_name": self.table_name, "data": df.to_dict(orient="records")}]
            logger.info(f"Loaded CSV {self.file_path} as table {self.table_name} with {len(df)} records (status=0)")
            return data
        except Exception as e:
            logger.error(f"Failed to load CSV file {self.file_path}: {traceback.format_exc()} (status=1)")
            raise

class APIDataSourceLoader(DataSourceLoader):
    """Loader for API data, mapping JSON to a single table."""
    
    def __init__(self, api_url: str, table_name: str):
        try:
            if not api_url or not table_name:
                logger.error("API URL or table name cannot be empty")
                raise ValueError("api_url and table_name must be non-empty")
            self.api_url = api_url
            self.table_name = table_name
            logger.debug(f"Initialized APIDataSourceLoader for {api_url} with table {table_name}")
        except Exception as e:
            logger.error(f"Initialization failed: {traceback.format_exc()}")
            raise

    def load_data(self) -> List[Dict[str, Any]]:
        try:
            logger.debug(f"Fetching data from API: {self.api_url}")
            response = requests.get(self.api_url)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list):
                data = data.get('data', []) or [data]
            if not data:
                logger.warning(f"No data returned from API {self.api_url} (status=1)")
                return []
            logger.debug(f"Loaded {len(data)} records from API {self.api_url}")
            return [{"table_name": self.table_name, "data": data}]
        except requests.RequestException as e:
            logger.error(f"Failed to load API data from {self.api_url}: {traceback.format_exc()} (status=1)")
            raise
        finally:
            logger.info(f"API load completed for {self.api_url} (status=0)")


class OpticalFailureDataSourceLoader(DataSourceLoader):
    """Loader for optical failure data from MongoDB."""
    
    def __init__(self, table_name: str = "事件监控-光模块故障表"):
        try:
            self.table_name = table_name
            logger.debug(f"Initialized OpticalFailureDataSourceLoader with table {table_name}")
        except Exception as e:
            logger.error(f"Initialization failed: {traceback.format_exc()}")
            raise

    def load_data(self) -> List[Dict[str, Any]]:
        try:
            logger.debug("Loading optical failure data from MongoDB")
            # 导入查询函数
            from backend.dataSources.optical_failure import query_event_monitor_demo
            
            # 获取数据
            data = query_event_monitor_demo()
            
            if not data:
                logger.warning("No optical failure data retrieved (status=1)")
                return []
            
            logger.debug(f"Loaded {len(data)} records from MongoDB")
            return [{"table_name": self.table_name, "data": data}]
        except Exception as e:
            logger.error(f"Failed to load optical failure data: {traceback.format_exc()} (status=1)")
            raise
        finally:
            logger.info("Optical failure data load completed (status=0)")


class OpticalModuleInventoryDataSourceLoader(DataSourceLoader):
    """Loader for optical module inventory data from API."""
    
    def __init__(self, table_name: str = "光模块在线总数量统计表"):
        try:
            self.table_name = table_name
            logger.debug(f"Initialized OpticalModuleInventoryDataSourceLoader with table {table_name}")
        except Exception as e:
            logger.error(f"Initialization failed: {traceback.format_exc()}")
            raise

    def load_data(self) -> List[Dict[str, Any]]:
        try:
            logger.debug("Loading optical module inventory data from API")
            # 导入查询函数
            from backend.dataSources.optical_module_inventory import query_optical_module_inventory
            
            # 获取数据
            data = query_optical_module_inventory()
            
            if not data:
                logger.warning("No optical module inventory data retrieved (status=1)")
                return []
            
            logger.debug(f"Loaded {len(data)} records from API")
            return [{"table_name": self.table_name, "data": data}]
        except Exception as e:
            logger.error(f"Failed to load optical module inventory data: {traceback.format_exc()} (status=1)")
            raise
        finally:
            logger.info("Optical module inventory data load completed (status=0)")


class RoceEventDataSourceLoader(DataSourceLoader):
    """Loader for ROCE event data from MongoDB."""
    
    def __init__(self, table_name: str = "ROCE网络事件-光模块故障表"):
        try:
            self.table_name = table_name
            logger.debug(f"Initialized RoceEventDataSourceLoader with table {table_name}")
        except Exception as e:
            logger.error(f"Initialization failed: {traceback.format_exc()}")
            raise

    def load_data(self) -> List[Dict[str, Any]]:
        try:
            logger.debug("Loading ROCE event data from MongoDB")
            # 导入查询函数
            from backend.dataSources.roce_event import query_roce_network_event_demo
            
            # 获取数据
            data = query_roce_network_event_demo()
            
            if not data:
                logger.warning("No ROCE event data retrieved (status=1)")
                return []
            
            logger.debug(f"Loaded {len(data)} records from MongoDB")
            return [{"table_name": self.table_name, "data": data}]
        except Exception as e:
            logger.error(f"Failed to load ROCE event data: {traceback.format_exc()} (status=1)")
            raise
        finally:
            logger.info("ROCE event data load completed (status=0)")
