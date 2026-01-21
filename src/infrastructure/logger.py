import logging
import sqlite3
import sys
from datetime import datetime
from typing import Optional

# Getting configuration from here (Dependency Injection)
from src.config.settings import settings

def write_log_db(level: str, message: str, timestamp: Optional[str] = None) -> None:
    """
    Writes logs to SQLite database.
    If database error occurs, writes to 'bot_logs_fallback.txt'.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    conn = None
    try:
        # Get DB path from Settings
        conn = sqlite3.connect(settings.LOG_DB_PATH)
        cur = conn.cursor()
        
        # Create table if not exists
        cur.execute(
            """CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                level TEXT,
                message TEXT
            )"""
        )
        
        # Save log
        cur.execute(
            "INSERT INTO logs (timestamp, level, message) VALUES (?, ?, ?)",
            (timestamp, level, message)
        )
        conn.commit()
        
    except Exception:
        # Fallback in case of DB Write error (Write to file)
        try:
            with open('bot_logs_fallback.txt', 'a', encoding='utf-8') as f:
                f.write(f"{timestamp} | {level} | {message}\n")
        except Exception:
            pass # If we can't write to file either, nothing to do, pass silently.
            
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

class DBHandler(logging.Handler):
    """
    Bridge connecting Python's standard logging module with our SQLite function.
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            write_log_db(record.levelname, msg)
        except Exception:
            self.handleError(record)

def setup_logger(name: str = 'ARASTA') -> logging.Logger:
    """
    Configures logger and returns instance.
    Sets up to log to both Console and Database.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # If handlers already exist, don't add again (For Multithreading/Restart cases)
    if not logger.handlers:
        # Common Format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 1. Console Handler (Colored/Standard output to screen)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 2. Database Handler (Save to SQLite)
        db_handler = DBHandler()
        db_handler.setLevel(logging.INFO)
        db_handler.setFormatter(formatter)
        logger.addHandler(db_handler)
    
    return logger

# Singleton Logger Instance
# In other files: from src.infrastructure.logger import logger
logger = setup_logger()