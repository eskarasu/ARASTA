from datetime import datetime
from collections import deque
from typing import Tuple, List, Optional
from src.config.settings import settings
from src.infrastructure.logger import logger

class EnhancedRiskManager:
    """
    Enhanced Risk Management Service.
    Manages balance, stop-loss rules, position sizing, 
    and short/long permissions.
    """
    
    def __init__(self):
        # Get constants from settings
        self.daily_loss_limit = settings.DAILY_LOSS_LIMIT
        self.max_consecutive_losses = settings.RISK_MANAGER_MAX_LOSSES
        
        # State variables
        self.daily_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self.last_reset = datetime.now().date()
        self.daily_trades: int = 0
        self.daily_trade_limit: int = 999  # Daily maximum trade count
        
        # Drawdown tracking
        self.drawdown_history = deque(maxlen=100)
        self.max_drawdown: float = 0.0
        self.peak_value: float = settings.SIMULATED_BALANCE
        self.current_balance: float = settings.SIMULATED_BALANCE
        
        # Statistics
        self.trade_history_list: List[float] = []
        self.short_trades: int = 0
        self.short_wins: int = 0
        self.long_trades: int = 0
        self.long_wins: int = 0

    def check_daily_reset(self) -> None:
        """Reset counters if day changed"""
        today = datetime.now().date()
        if today != self.last_reset:
            self.daily_pnl = 0.0
            self.consecutive_losses = 0
            self.daily_trades = 0
            self.last_reset = today
            logger.info("📅 Daily risk metrics reset.")

    def record_trade(self, profit_pct: float, is_short: bool = False) -> None:
        """Records a completed trade and updates statistics"""
        self.check_daily_reset()
        
        # PnL and Trade count update
        self.daily_pnl += profit_pct
        self.daily_trades += 1
        
        # History list update
        self.trade_history_list.append(profit_pct)
        if len(self.trade_history_list) > 100:
            self.trade_history_list.pop(0)
        
        # Consecutive loss counter
        if profit_pct < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            
        # Short/Long statistics
        if is_short:
            self.short_trades += 1
            if profit_pct > 0: self.short_wins += 1
        else:
            self.long_trades += 1
            if profit_pct > 0: self.long_wins += 1

        # Drawdown calculation
        if self.current_balance > self.peak_value:
            self.peak_value = self.current_balance
            
        if self.peak_value > 0:
            current_drawdown = (self.current_balance - self.peak_value) / self.peak_value * 100
            self.drawdown_history.append(current_drawdown)
            if current_drawdown < self.max_drawdown:
                self.max_drawdown = current_drawdown

    def calculate_position_size(self, symbol: str, current_balance: float, volatility: float = 5.0) -> float:
        """
        Calculates dynamic position size based on Kelly Criterion and Volatility.
        """
        self.current_balance = current_balance # Update balance
        
        # Base position size (From settings)
        base_size = settings.INVESTMENT_AMOUNT
        
        # Volatility factor (High volatility -> Smaller position)
        # As volatility increases (e.g., 20), factor decreases.
        volatility_factor = max(0.3, 1.0 - (volatility / 20.0))
        
        # Kelly-like Approach (Scaling by win rate)
        win_rate = self.get_win_rate()
        kelly_factor = max(0.5, min(1.5, win_rate * 2)) # Simplified
        
        adjusted_size = base_size * volatility_factor * kelly_factor
        
        # Maximum limit check (Cannot exceed X% of portfolio)
        max_size = current_balance * (settings.MAX_POSITION_SIZE_PCT / 100)
        
        # Minimum trade amount check
        final_size = max(settings.MIN_INVESTMENT, min(adjusted_size, max_size))
        
        return round(final_size, 2)

    def calculate_short_position_size(self, current_balance: float, volatility: float) -> float:
        """Special position calculation for Short trades (Generally more conservative)"""
        base_size = self.calculate_position_size("SHORT", current_balance, volatility)
        
        # Reduce size if short success is low
        short_wr = self.get_short_win_rate()
        if short_wr < 0.4:
            base_size *= 0.6
            
        return round(base_size, 2)

    def get_win_rate(self) -> float:
        """Returns overall win rate (0.0 - 1.0)"""
        if not self.trade_history_list:
            return 0.65  # Default optimistic prediction if no history
            
        wins = sum(1 for trade in self.trade_history_list if trade > 0)
        return wins / len(self.trade_history_list)

    def get_short_win_rate(self) -> float:
        """Win rate of short trades"""
        if self.short_trades == 0:
            return 0.5
        return self.short_wins / self.short_trades

    def should_allow_short(self) -> Tuple[bool, str]:
        """Is short trading allowed?"""
        if not settings.SHORT_SELL:
            return False, "Short trading settings disabled"
            
        # Check performance of last 5 short trades
        if self.short_trades >= 5:
            short_wr = self.get_short_win_rate()
            if short_wr < 0:
                return False, f"Short success rate too low: %{short_wr*100:.1f}"
                
        return True, ""

    def should_stop_trading(self) -> Tuple[bool, str]:
        """
        Checks if trading should be stopped.
        (Circuit Breaker Logic)
        """
        self.check_daily_reset()
        
        # 1. Daily Loss Limit
        if self.daily_pnl <= self.daily_loss_limit:
            return True, f"Daily loss limit exceeded: %{self.daily_pnl:.2f} <= %{self.daily_loss_limit}"
        
        # 2. Consecutive Loss Limit
        if self.consecutive_losses >= self.max_consecutive_losses:
            return True, f"{self.consecutive_losses} trades lost consecutively."
            
        # 3. Daily Trade Limit
        if self.daily_trades >= self.daily_trade_limit:
            return True, f"Daily trade limit ({self.daily_trade_limit}) reached."
            
        return False, ""