import sqlite3
import os
from typing import Dict
from src.infrastructure.logger import logger
from src.config.settings import settings

class DatabaseAdapter:
    """
    Saves only executed trades (Trade Log) and current indicator scores.
    """
    def __init__(self, db_name="backtest_trades.db"):
        self.db_name = db_name
        self._init_db()

    def _init_db(self):
        """Creates table"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # You can create table clean (drop) or use if not exists.
            # You may need to delete old db file due to schema change.
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    position_type TEXT,
                    entry_timestamp DATETIME,
                    exit_timestamp DATETIME,
                    entry_price REAL,
                    exit_price REAL,
                    pnl_usdt REAL,
                    roi_percent REAL,
                    exit_reason TEXT DEFAULT 'OPEN',
                    status TEXT DEFAULT 'OPEN',
                    
                    -- Indicator Score Details (At Entry)
                    score_total REAL,
                    score_rsi REAL,
                    score_macd REAL,
                    score_stoch REAL,
                    score_williams REAL,
                    score_adx REAL,
                    score_ema REAL,
                    score_bb REAL,
                    
                    -- New Added Components
                    score_sentiment REAL DEFAULT 0,
                    score_trend_health REAL DEFAULT 0,
                    score_volume REAL DEFAULT 0,
                    score_patterns REAL DEFAULT 0,
                    score_bonus REAL DEFAULT 0,

                    -- Performance Metrics (Max/Min ROI)
                    max_profit_pct REAL DEFAULT 0,
                    max_loss_pct REAL DEFAULT 0
                )
            ''')
            
            # Add status column if not in old table (Migration)
            try:
                cursor.execute("ALTER TABLE trades ADD COLUMN status TEXT DEFAULT 'CLOSED'")
            except:
                pass # Do not error if column exists
            
            # Migration for new columns (Compatibility with old DB)
            for col in ['score_sentiment', 'score_trend_health', 'score_volume', 'score_patterns', 'score_bonus', 'max_profit_pct', 'max_loss_pct']:
                try: cursor.execute(f"ALTER TABLE trades ADD COLUMN {col} REAL DEFAULT 0")
                except: pass
                
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Database initialization error: {e}")

    def save_trade(self, trade_data: Dict):
        """Writes a closed trade to database"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades (
                    symbol, position_type, entry_timestamp, exit_timestamp,
                    entry_price, exit_price, pnl_usdt, roi_percent, exit_reason,
                    score_total, score_rsi, score_macd, score_stoch, 
                    score_williams, score_adx, score_ema, score_bb,
                    score_sentiment, score_trend_health, score_volume,
                    score_patterns, score_bonus, max_profit_pct, max_loss_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data['symbol'], trade_data['type'], trade_data['entry_time'], trade_data['exit_time'],
                trade_data['entry_price'], trade_data['exit_price'], trade_data['pnl'], trade_data['roi'], trade_data['reason'],
                
                # Indicator Scores (0.0 if missing)
                trade_data.get('score_total', 0),
                trade_data.get('score_rsi', 0),
                trade_data.get('score_macd', 0),
                trade_data.get('score_stoch', 0),
                trade_data.get('score_williams', 0),
                trade_data.get('score_adx', 0),
                trade_data.get('score_ema', 0),
                trade_data.get('score_bb', 0),
                trade_data.get('score_sentiment', 0),
                trade_data.get('score_trend_health', 0),
                trade_data.get('score_volume', 0),
                trade_data.get('score_patterns', 0),
                trade_data.get('score_bonus', 0),
                trade_data.get('max_profit_pct', 0),
                trade_data.get('max_loss_pct', 0)
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Trade save error: {e}")

    def log_open_trade(self, trade_data: Dict) -> int:
        """
        Saves a newly opened trade to database and returns its ID.
        """
        trade_id = -1
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades (
                    symbol, position_type, entry_timestamp, entry_price, 
                    status, exit_reason,
                    score_total, score_rsi, score_macd, score_stoch, 
                    score_williams, score_adx, score_ema, score_bb,
                    score_sentiment, score_trend_health, score_volume,
                    score_patterns, score_bonus
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data['symbol'], trade_data['type'], trade_data['entry_time'], trade_data['entry_price'],
                'OPEN', 'Monitoring',
                
                trade_data.get('score_total', 0),
                trade_data.get('score_rsi', 0),
                trade_data.get('score_macd', 0),
                trade_data.get('score_stoch', 0),
                trade_data.get('score_williams', 0),
                trade_data.get('score_adx', 0),
                trade_data.get('score_ema', 0),
                trade_data.get('score_bb', 0),
                trade_data.get('score_sentiment', 0),
                trade_data.get('score_trend_health', 0),
                trade_data.get('score_volume', 0),
                trade_data.get('score_patterns', 0),
                trade_data.get('score_bonus', 0)
            ))
            
            trade_id = cursor.lastrowid
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Open trade save error: {e}")
        
        return trade_id

    def update_trade_exit(self, trade_id: int, exit_data: Dict):
        """
        Updates open trade as closed.
        """
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE trades 
                SET exit_timestamp = ?, exit_price = ?, pnl_usdt = ?, roi_percent = ?, exit_reason = ?, status = 'CLOSED',
                    max_profit_pct = ?, max_loss_pct = ?
                WHERE id = ?
            ''', (
                exit_data['exit_time'], exit_data['exit_price'], exit_data['pnl'], 
                exit_data['roi'], exit_data['reason'], 
                exit_data.get('max_profit_pct', 0), exit_data.get('max_loss_pct', 0),
                trade_id
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Trade update error: {e}")

    def update_trade_performance(self, trade_id: int, max_profit: float, max_loss: float):
        """Updates performance data (Max Profit/Loss) of open trade"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE trades 
                SET max_profit_pct = ?, max_loss_pct = ?
                WHERE id = ?
            ''', (max_profit, max_loss, trade_id))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Performance update error: {e}")

class LiveDatabaseAdapter(DatabaseAdapter):
    """
    Specialized database adapter for live trades.
    """
    def __init__(self):
        super().__init__(settings.LIVE_TRADES_DB_PATH)