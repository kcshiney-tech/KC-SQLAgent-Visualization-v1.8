# database_builder.py
import sqlite3
import logging
import re
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Dict, Any, Sequence
from datetime import datetime, date
from backend.data_loader import DataSourceLoader,OpticalFailureDataSourceLoader,OpticalModuleInventoryDataSourceLoader,RoceEventDataSourceLoader,NetworkDeviceInventoryDataSourceLoader,NetworkDeviceFailureDataSourceLoader

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("database_builder.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

class DatabaseBuilder:
    """Builder for creating or updating SQLite databases from data sources, supporting Unicode names."""
    
    def __init__(self, db_path: str):
        """
        Initialize DatabaseBuilder.

        Args:
            db_path (str): Path to the SQLite database file.

        Raises:
            ValueError: If db_path is empty or invalid.
        """
        if not db_path:
            logger.error("Database path cannot be empty (status=1)")
            raise ValueError("db_path must be non-empty")
        self.db_path = db_path
        logger.debug(f"Initialized DatabaseBuilder with db_path: {db_path}")

    def sanitize_name(self, name: str, used_names: set) -> str:
        """
        Sanitize table or column name to be SQL-compatible, preserving Chinese characters.

        Args:
            name (str): Original name.
            used_names (set): Set of already used sanitized names.

        Returns:
            str: Sanitized, unique name (max 63 chars for SQLite).
        """
        if not name.strip():
            base_name = "name"
        else:
            # Replace specific invalid characters, preserve Chinese
            base_name = re.sub(r'[=\/\\;,"\'\s]', '_', name)
            base_name = re.sub(r'_+', '_', base_name).strip('_')
            if not base_name:
                base_name = "name"
        # Ensure length and uniqueness
        final_name = base_name[:60]  # SQLite name limit ~63 chars
        counter = 1
        while final_name in used_names:
            final_name = f"{base_name[:55]}_{counter}"
            counter += 1
        used_names.add(final_name)
        logger.debug(f"Sanitized name: {name} -> {final_name}")
        return final_name

    def quote_name(self, name: str) -> str:
        """
        Quote name for SQLite to handle special characters and Chinese.

        Args:
            name (str): Name to quote.

        Returns:
            str: Quoted name (e.g., "name").
        """
        return f'"{name}"'

    def infer_column_type(self, value: Any) -> str:
        """
        Infer SQLite column type from value.

        Args:
            value: Sample value to infer type from.

        Returns:
            str: SQLite type (TEXT, INTEGER, REAL).
        """
        if isinstance(value, (int, bool)) and not isinstance(value, bool):
            return "INTEGER"
        elif isinstance(value, bool):
            return "INTEGER"  # SQLite stores booleans as 0/1
        elif isinstance(value, float):
            return "REAL"
        elif isinstance(value, (datetime, date)):
            return "TEXT"
        elif isinstance(value, str):
            for fmt in [
                "%Y/%m/%d", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f"
            ]:
                try:
                    datetime.strptime(value, fmt)
                    return "TEXT"
                except ValueError:
                    continue
            return "TEXT"
        else:
            return "TEXT"

    def build_database(self, data_sources: Sequence[DataSourceLoader], rebuild: bool = True) -> Dict[str, Any]:
        """
        Build or update SQLite database from data sources.

        Args:
            data_sources (List[DataSourceLoader]): List of data source loaders.
            rebuild (bool): If True, drop and recreate tables; if False, append data.

        Returns:
            Dict[str, Any]: {'status': int, 'errors': List[str]} where status is 0 (success) or 1 (failure),
                            and errors lists any issues encountered.

        Raises:
            sqlite3.Error: If critical database operations fail.
        """
        errors = []
        status = 0
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Drop invalid table ______
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='______'")
            if cursor.fetchone():
                cursor.execute("DROP TABLE ______")
                logger.info("Dropped invalid table ______")

            for source in data_sources:
                logger.debug(f"Processing data source: {source}")
                try:
                    for table_data in source.load_data():
                        original_table_name = table_data["table_name"]
                        table_name = original_table_name  # Use original name directly
                        quoted_table_name = self.quote_name(table_name)
                        records = table_data["data"]
                        if not records:
                            logger.warning(f"No data for table {table_name} (status=1)")
                            errors.append(f"No data for table {table_name}")
                            status = 1
                            continue

                        # Infer schema
                        sample_record = records[0]
                        columns = []
                        used_names = set()
                        primary_key = None
                        logger.debug(f"Sample record columns: {list(sample_record.keys())}")
                        for key, value in sample_record.items():
                            col_name = self.sanitize_name(key, used_names)
                            quoted_col_name = self.quote_name(col_name)
                            col_type = self.infer_column_type(value)
                            if col_name.lower() in ('id', '_id'):
                                primary_key = col_name
                                col_def = f"{quoted_col_name} {col_type} PRIMARY KEY"
                            else:
                                col_def = f"{quoted_col_name} {col_type}"
                            columns.append(col_def)
                        columns_def = ", ".join(columns)

                        # Validate schema
                        if len(used_names) != len(sample_record):
                            logger.error(f"Duplicate column names detected in {table_name}: {sample_record.keys()} (status=1)")
                            errors.append(f"Duplicate column names in {table_name}")
                            status = 1
                            continue

                        # Create or update table
                        try:
                            if rebuild:
                                cursor.execute(f"DROP TABLE IF EXISTS {quoted_table_name}")
                                cursor.execute(f"CREATE TABLE {quoted_table_name} ({columns_def})")
                                logger.info(f"Rebuilt table {table_name} from {original_table_name}")
                            else:
                                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                                if cursor.fetchone() is None:
                                    cursor.execute(f"CREATE TABLE {quoted_table_name} ({columns_def})")
                                    logger.info(f"Created new table {table_name} from {original_table_name}")
                                else:
                                    logger.info(f"Appending to existing table {table_name}")
                        except sqlite3.Error as e:
                            logger.error(f"Failed to create table {table_name}: {e} (status=1)")
                            errors.append(f"Table creation failed for {table_name}: {e}")
                            status = 1
                            continue

                        # Prepare batch insert
                        quoted_columns = ", ".join([self.quote_name(self.sanitize_name(k, set())) for k in sample_record.keys()])
                        placeholders = ", ".join(["?" for _ in sample_record])
                        values_list = [
                            [record.get(col, None) if not isinstance(record.get(col), (datetime, date))
                             else record.get(col).isoformat() for col in sample_record.keys()]
                            for record in records
                        ]

                        # Batch insert
                        try:
                            cursor.executemany(
                                f"INSERT OR IGNORE INTO {quoted_table_name} ({quoted_columns}) VALUES ({placeholders})",
                                values_list
                            )
                            logger.info(f"Inserted {len(records)} records into table {table_name}")
                        except sqlite3.Error as e:
                            logger.error(f"Failed to insert records into {table_name}: {e} (status=1)")
                            errors.append(f"Record insertion failed for {table_name}: {e}")
                            status = 1
                            continue

                except Exception as e:
                    logger.error(f"Failed to process data source {source}: {e} (status=1)")
                    errors.append(f"Data source processing failed: {e}")
                    status = 1
                    continue

            conn.commit()
            logger.info(f"Database built/updated at {self.db_path} (rebuild={rebuild}, status={status})")
            return {"status": status, "errors": errors}

        except sqlite3.Error as e:
            logger.error(f"Critical database operation failed: {e} (status=1)")
            errors.append(f"Critical database error: {e}")
            return {"status": 1, "errors": errors}
        finally:
            conn.close()

    def drop_tables(self, table_names: List[str]) -> Dict[str, Any]:
        """
        删除数据库中的指定数据表

        Args:
            table_names (List[str]): 要删除的数据表名称列表

        Returns:
            Dict[str, Any]: {'status': int, 'errors': List[str]} where status is 0 (success) or 1 (failure),
                            and errors lists any issues encountered.

        Raises:
            sqlite3.Error: If critical database operations fail.
        """
        errors = []
        status = 0
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for table_name in table_names:
                try:
                    # 检查表是否存在
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
                    if cursor.fetchone():
                        # 删除表
                        cursor.execute(f"DROP TABLE {self.quote_name(table_name)}")
                        logger.info(f"Dropped table {table_name}")
                    else:
                        logger.warning(f"Table {table_name} does not exist")
                except sqlite3.Error as e:
                    logger.error(f"Failed to drop table {table_name}: {e} (status=1)")
                    errors.append(f"Failed to drop table {table_name}: {e}")
                    status = 1
                    continue

            conn.commit()
            logger.info(f"Dropped tables: {table_names} (status={status})")
            return {"status": status, "errors": errors}

        except sqlite3.Error as e:
            logger.error(f"Critical database operation failed: {e} (status=1)")
            errors.append(f"Critical database error: {e}")
            return {"status": 1, "errors": errors}
        finally:
            conn.close()

if __name__ == "__main__":
    from data_loader import ExcelDataSourceLoader, APIDataSourceLoader, OpticalFailureDataSourceLoader, OpticalModuleInventoryDataSourceLoader, RoceEventDataSourceLoader, NetworkDeviceInventoryDataSourceLoader, NetworkDeviceFailureDataSourceLoader, NOCOpticalModuleFullDataSourceLoader
    data_sources = [
        # ExcelDataSourceLoader("20250813光模块分析.xlsx", sheets=[("工作表6", "光模块故障表")]),
        # ExcelDataSourceLoader("20250813光模块分析.xlsx", sheets=[("工作表5", "光模块故障表"),("工作表6", "光模块故障表")])
        # APIDataSourceLoader("https://jsonplaceholder.typicode.com/users", "users"
        OpticalFailureDataSourceLoader(),
        # OpticalModuleInventoryDataSourceLoader(),  # 使用新的NOC全量数据源替代
        NOCOpticalModuleFullDataSourceLoader(),
        RoceEventDataSourceLoader(),
        NetworkDeviceInventoryDataSourceLoader(),
        NetworkDeviceFailureDataSourceLoader()
    ]
    builder = DatabaseBuilder("custom_database.db")
    result = builder.build_database(data_sources, rebuild=True)
    print(f"Database build completed with status: {result['status']}, errors: {result['errors']}")
