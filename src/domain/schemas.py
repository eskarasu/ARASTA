from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List, Union
from enum import Enum

# --- ENUM Classes (Fixed Options) ---
class PositionType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class TrendState(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    STRONG_BULLISH = "STRONG_BULLISH"
    STRONG_BEARISH = "STRONG_BEARISH"

# --- DATA STRUCTURES (Dataclasses) ---

@dataclass
class MarketData:
    """Single candle data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class TechnicalAnalysisResult:
    """Result from Technical Analysis Engine"""
    symbol: str
    current_price: float
    timestamp: float
    
    # Basic Indicators
    rsi: float
    macd: float
    macd_signal: float
    adx: float
    
    # Scores
    trend_score: int
    momentum_score: float
    volatility: float  # NATR veya ATR
    volume_ratio: float
    
    # Patterns (Kept as Dict because dynamic)
    patterns: Dict[str, bool] = field(default_factory=dict)
    
    # Calculated Final Scores
    long_score: float = 0.0
    short_score: float = 0.0

@dataclass
class TradeOpportunity:
    """Trade opportunity approved by strategy"""
    symbol: str
    signal_type: PositionType  # LONG veya SHORT
    entry_price: float
    score: float
    
    # Risk Management Parameters
    suggested_stop_loss: float
    suggested_take_profit: float
    volatility: float
    
    # Extra analysis data (For logging)
    analysis_data: Optional[TechnicalAnalysisResult] = None
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

@dataclass
class Position:
    """Represents an open position (Item in portfolio)"""
    symbol: str
    position_type: PositionType
    
    # Entry Info
    entry_price: float
    amount: float            # Coin amount
    entry_time: datetime
    initial_investment: float # USDT amount
    leverage: int = 1
    
    # Current Status
    margin_used: float = 0.0
    liquidation_price: float = 0.0
    
    # Targets
    take_profit_pct: float   # Örn: 5.0 (%)
    stop_loss_pct: float     # Örn: -2.0 (%)
    
    # Trailing Stop
    trailing_stop_price: Optional[float] = None
    highest_price: Optional[float] = None  # Highest seen for Long
    lowest_price: Optional[float] = None   # Lowest seen for Short
    
    # Binance Filter Info (Required when selling)
    filter_info: Dict = field(default_factory=dict)
    
    retry_count: int = 0

@dataclass
class TradeLog:
    """Completed trade to be saved to history"""
    symbol: str
    position_type: str
    entry_price: float
    exit_price: float
    quantity: float
    
    profit_pct: float
    profit_amount: float
    
    entry_time: datetime
    exit_time: datetime
    hold_time_hours: float
    
    exit_reason: str
    fees_paid: float = 0.0
    leverage: int = 1