import sqlite3
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("test_database.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

def test_display_database(db_path: str) -> None:
    """
    Display all tables and their data from the SQLite database.
    
    Args:
        db_path (str): Path to the SQLite database file.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        logger.info(f"Found {len(tables)} tables in {db_path}")
        
        for table in tables:
            table_name = table[0]
            logger.info(f"\nTable: {table_name}")
            
            # Get column names
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = [col[1] for col in cursor.fetchall()]
            logger.info(f"Columns: {', '.join(columns)}")
            
            # Fetch all rows
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            logger.info(f"Total rows: {len(rows)}")
            
            # Display rows
            for row in rows:
                row_data = dict(zip(columns, row))
                logger.debug(f"Row: {row_data}")
                
        logger.info("Database display completed (status=0)")
        
    except sqlite3.Error as e:
        logger.error(f"Failed to display database: {e} (status=1)")
    finally:
        conn.close()

if __name__ == "__main__":
    test_display_database("custom_database.db")