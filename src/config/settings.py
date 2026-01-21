import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Settings:
    """
    Centralizes all bot configuration.
    Manages environment variables (.env) and constants.
    """

    # --- 1. API and System Settings ---
    API_KEY = os.getenv('BINANCE_API_KEY')
    API_SECRET = os.getenv('BINANCE_SECRET_KEY')
    LOG_DB_PATH = os.getenv('BOT_LOG_DB', 'enhanced_daily_trader_logs.db')
    LIVE_TRADES_DB_PATH = os.getenv('LIVE_TRADES_DB_PATH', 'live_trades.db')
    SENTIMENT_HISTORY_FILE = 'sentiment_history.json'
    
    # Test and Determinism (Reproducibility)
    DETERMINISTIC = os.getenv('DETERMINISTIC', 'true').strip().lower() in ('1', 'true', 'yes')
    FORCE_MARKET = os.getenv('FORCE_MARKET', '').strip().upper()

    # --- 2. Trading Mode (Spot / Futures) ---
    LIVE_TRADING = os.getenv('LIVE_TRADING', 'false').strip().lower() in ('1', 'true', 'yes')
    SIMULATED_BALANCE = float(os.getenv('SIMULATED_BALANCE', '1000.0'))
    
    USE_FUTURES = os.getenv('USE_FUTURES', 'false').strip().lower() in ('1', 'true', 'yes')
    LEVERAGE = int(os.getenv('LEVERAGE', '3'))
    MARGIN_TYPE = os.getenv('MARGIN_TYPE', 'ISOLATED')
    POSITION_SIDE = os.getenv('POSITION_SIDE', 'BOTH')
    ENABLE_FUTURES_TESTNET = os.getenv('ENABLE_FUTURES_TESTNET', 'false').strip().lower() in ('1', 'true', 'yes')

    # --- 3. Strategy Parameters ---
    # Short Trade Setting
    SHORT_SELL = os.getenv('SHORT_SELL', 'false').strip().lower() in ('1', 'true', 'yes')
    SKIP_BTC_TREND_FILTER = True
    FOLLOW_MARKET_REGIME = os.getenv('FOLLOW_MARKET_REGIME', 'true').strip().lower() in ('1', 'true', 'yes')
    # Scoring Thresholds (Main Entry)
    MIN_SCORE_REQUIREMENT = 7.0  # Balanced Threshold (7.0 is an ideal filter with the new scoring system)
    SHORT_SCORE_THRESHOLD = float(os.getenv('SHORT_SCORE_THRESHOLD', '-10.0'))
    
    # --- NEW ADDED: TREND HEALTH / SMART ENTRY SETTINGS ---
    # These settings are required for calculate_trend_health function in strategies.py.
    SMART_ENTRY_BB_THRESHOLD: float = 0.95        # Bollinger Band ceiling proximity limit
    SMART_ENTRY_RSI_THRESHOLD: float = 75.0       # RSI Overbought limit
    FRESH_SIGNAL_MIN_MOMENTUM: float = 0.5        # Minimum price momentum
    FRESH_SIGNAL_VOLUME_RATIO: float = 1.2        # Minimum volume increase ratio
    SMART_ENTRY_MA_SAFE_MAX: float = 5.5          # Maximum distance from moving average (%)
    # ---------------------------------------------------------

    # Time and Periods
    LOOKUP_INTERVAL = '1h'
    ANALYSIS_PERIOD = 100
    PRICE_CHECK_INTERVAL = 20  # Seconds (180s is too long for leveraged trading, reduced to 20s)
    RECENTLY_SOLD_COOLDOWN = 60 * 60  # 1 Hour
    
    # Holding Periods
    MAX_HOLD_TIME_DAYS =  5
    
    # --- 4. Position and Balance Management ---
    INVESTMENT_AMOUNT = 10
    MIN_INVESTMENT = 10
    MAX_HOLDINGS = 50
    
    # Dynamic Settings
    DYNAMIC_PROFIT_TARGET = True
    DYNAMIC_STOP_LOSS = True

    # --- 5. Risk Management ---
    RISK_MANAGER_MAX_LOSSES = 2
    DAILY_LOSS_LIMIT = -20.0
    MAX_POSITION_SIZE_PCT = 25.0
    VOLATILITY_THRESHOLD = 15.0
    MAX_ALLOWED_LOSS_ON_EXIT = float(os.getenv('MAX_ALLOWED_LOSS_ON_EXIT', '10')) # 4% price change for 10x leverage (40% ROI Stop Loss)
    
    # --- Take Profit Fine Tuning ---
    MIN_PROFIT_FOR_INDICATOR_EXIT = 10  # Min 3.5% profit for indicator exits (Prevents early selling)
    RSI_EXIT_THRESHOLD_LONG = 82.0       # Expect RSI 82 instead of 78 (For more profit)
    RSI_EXIT_THRESHOLD_SHORT = 18.0      # Expect RSI 18 instead of 25
    TRADE_FEE = 0.001

    # --- 6. Symbol Lists ---
    SYMBOL_BLACKLIST = {
        'BUSDUSDT', 'TUSDUSDT', 'USDCUSDT', 'DAIUSDT', 'PAXUSDT', 'USTCUSDT', 'USDPUSDT',
        'PAXGUSDT', 'LUNAUSDT', 'YFIIUSDT', 'COCOSUSDT', 'FDUSDUSDT', 'USDEUSDT', 
        'AIUSDT', 'XVSUSDT'
    }

    PREFERRED_SYMBOLS = {
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'AVAXUSDT', 'MATICUSDT',
        'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'UNIUSDT', 'AAVEUSDT', 'XRPUSDT',
        'ALGOUSDT', 'ATOMUSDT', 'NEARUSDT', 'FTMUSDT', 'SANDUSDT', 'MANAUSDT'
    }

    @staticmethod
    def validate():
        """Validates critical settings"""
        if not Settings.API_KEY or not Settings.API_SECRET:
            raise ValueError("❌ ERROR: API keys not found in .env file!")
        
        if Settings.LIVE_TRADING:
            print("⚠️ WARNING: Live trading mode (LIVE_TRADING) is active!")

# Creating instance for ease of use
settings = Settings()