import sqlite3
import pandas as pd
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("test_export_database.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

def export_database_to_xlsx(db_path: str, output_dir: str = "output") -> dict:
    """
    Export all tables from SQLite database to an Excel file with one sheet per table.

    Args:
        db_path (str): Path to the SQLite database file.
        output_dir (str): Directory to save the output Excel file.

    Returns:
        dict: {'status': int, 'file_path': str, 'errors': List[str]} where status is 0 (success) or 1 (failure),
              file_path is the path to the generated Excel file, and errors lists any issues.

    Raises:
        sqlite3.Error: If database operations fail.
        OSError: If file writing fails.
    """
    errors = []
    status = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"database_export_{timestamp}.xlsx")

    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Output directory ensured: {output_dir}")

        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"Found {len(tables)} tables in {db_path}: {tables}")

        if not tables:
            logger.warning("No tables found in database (status=1)")
            errors.append("No tables found in database")
            status = 1
            return {"status": status, "file_path": "", "errors": errors}

        # Create Excel writer
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            for table_name in tables:
                try:
                    # Fetch column names
                    cursor.execute(f"PRAGMA table_info('{table_name}');")
                    columns = [col[1] for col in cursor.fetchall()]
                    logger.debug(f"Columns in table {table_name}: {columns}")

                    # Fetch data
                    df = pd.read_sql_query(f"SELECT * FROM '{table_name}'", conn)
                    logger.debug(f"Read {len(df)} rows from table {table_name}")

                    # Write to Excel sheet
                    df.to_excel(writer, sheet_name=table_name, index=False)
                    logger.info(f"Exported table {table_name} to sheet in {output_file}")
                except Exception as e:
                    logger.error(f"Failed to export table {table_name}: {e} (status=1)")
                    errors.append(f"Failed to export table {table_name}: {e}")
                    status = 1
                    continue

        logger.info(f"Database exported to {output_file} (status={status})")
        return {"status": status, "file_path": output_file, "errors": errors}

    except sqlite3.Error as e:
        logger.error(f"Database operation failed: {e} (status=1)")
        errors.append(f"Database error: {e}")
        return {"status": 1, "file_path": "", "errors": errors}
    except OSError as e:
        logger.error(f"File operation failed: {e} (status=1)")
        errors.append(f"File error: {e}")
        return {"status": 1, "file_path": "", "errors": errors}
    finally:
        conn.close()

if __name__ == "__main__":
    # result = export_database_to_xlsx("custom_database.db")
    result = export_database_to_xlsx("test_network_device_failure.db")
    print(f"Export completed with status: {result['status']}, file: {result['file_path']}, errors: {result['errors']}")
