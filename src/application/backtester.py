import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import talib

# --- PROJECT MODULES ---
from src.config.settings import settings
from src.infrastructure.binance_api import BinanceAdapter
from src.infrastructure.analysis import AdvancedTechnicalAnalysis
from src.application.strategies import TradingStrategy
from src.infrastructure.logger import logger
from src.infrastructure.database_adapter import DatabaseAdapter
from src.infrastructure.sentiment_provider import SentimentProvider

class HistoricalSentiment:
    """
    Historical Market Regime Simulator for Backtest.
    Simulates BTC trend (EMA 50/200) in the tested date range.
    """
    def __init__(self, adapter: BinanceAdapter, start_str: str, end_str: str):
        self.adapter = adapter
        self.btc_df = pd.DataFrame()
        self.current_regime = 'NEUTRAL'
        self.btc_trend = 'NEUTRAL'
        self.daily_bias = 'NEUTRAL' # NEW
        self.fear_greed_index = 50 
        self.provider = SentimentProvider() # Real data provider
        
        # Prepare data
        self._prepare_btc_data(start_str, end_str)

    def _prepare_btc_data(self, start_str: str, end_str: str):
        """Downloads BTC data (4H) and calculates EMAs"""
        try:
            # Need more historical data (90 days) for EMA 300 (Daily 50)
            start_dt = datetime.strptime(start_str, "%Y-%m-%d") - timedelta(days=90)
            start_ts = int(start_dt.timestamp() * 1000)
            end_ts = int(datetime.strptime(end_str, "%Y-%m-%d").timestamp() * 1000)
            
            logger.info("⏳ Preparing BTC Trend Data for Backtest...")
            
            # Fetch BTC data in chunks
            all_klines = []
            current_start = start_ts
            
            while True:
                if current_start >= end_ts: break
                klines = self.adapter.client.futures_klines(
                    symbol='BTCUSDT', interval='4h', 
                    startTime=current_start, endTime=end_ts, limit=1000
                )
                if not klines: break
                all_klines.extend(klines)
                current_start = klines[-1][0] + 1
                time.sleep(0.1)

            if not all_klines:
                logger.warning("⚠️ BTC trend data could not be retrieved! Will run in neutral mode.")
                return

            df = pd.DataFrame(all_klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            df['close'] = pd.to_numeric(df['close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate EMA
            df['EMA_50'] = talib.EMA(df['close'], timeperiod=50)
            df['EMA_200'] = talib.EMA(df['close'], timeperiod=200)
            # NEW: Daily EMA 50 Simulation (300 candles in 4H chart ~ 50 days)
            df['EMA_300'] = talib.EMA(df['close'], timeperiod=300)
            
            df['RSI'] = talib.RSI(df['close'], timeperiod=14) # For Fear/Greed simulation
            
            self.btc_df = df.set_index('timestamp').sort_index()
            logger.info(f"✅ BTC Trend Data Ready: {len(df)} candles (4H)")
            
        except Exception as e:
            logger.error(f"BTC data preparation error: {e}")

    def update_for_time(self, current_time: datetime):
        """Updates regime based on time during backtest"""
        if self.btc_df.empty: return

        try:
            # Find latest BTC data before that moment (asof)
            # method='pad' -> backward fill (take nearest past data)
            idx = self.btc_df.index.get_indexer([current_time], method='pad')[0]
            if idx == -1: return # No data

            row = self.btc_df.iloc[idx]
            price = row['close']
            ema50 = row['EMA_50']
            ema200 = row['EMA_200']
            ema300 = row.get('EMA_300', np.nan) # Daily Trend Proxy
            rsi = row['RSI']
            
            if np.isnan(ema50) or np.isnan(ema200):
                self.regime = 'NEUTRAL'
            elif price > ema50 and ema50 > ema200:
                self.regime = 'BULL'
                self.btc_trend = 'BULLISH'
            elif price < ema50 and ema50 < ema200:
                self.regime = 'BEAR'
                self.btc_trend = 'BEARISH'
            else:
                self.regime = 'NEUTRAL'
                self.btc_trend = 'NEUTRAL'
            
            # --- DAILY TREND SIMULATION ---
            if not np.isnan(ema300):
                if price > ema300: self.daily_bias = 'BULLISH'
                else: self.daily_bias = 'BEARISH'
            else:
                self.daily_bias = 'NEUTRAL'
            
            # --- REAL FEAR & GREED DATA ---
            # Fetching that day's data via SentimentProvider.
            fng_data = self.provider.get_sentiment_at_date(current_time)
            
            if fng_data:
                self.fear_greed_index = fng_data['value']
                self.prediction_trend = fng_data['trend']
            else:
                # If no data (e.g. pre-2018), use RSI as Fallback
                if not np.isnan(rsi):
                    synthetic_fg = 50 + (rsi - 50) * 1.5
                    self.fear_greed_index = max(0, min(100, int(synthetic_fg)))
                    
                    # Simple RSI trend
                    try:
                        prev_rsi = self.btc_df.iloc[idx-3]['RSI']
                        if rsi < 30 and (rsi - prev_rsi) > 5: self.prediction_trend = 'RAPID_RECOVERY'
                        else: self.prediction_trend = 'NEUTRAL'
                    except: self.prediction_trend = 'NEUTRAL'
                
            self.calculate_score()
            
        except Exception:
            pass

    def calculate_score(self):
        score = 0.0
        if self.regime == 'BULL': score += 2.0
        elif self.regime == 'BEAR': score -= 2.0
        
        # Fear & Greed Score (Compatible with Strategies.py)
        # Use simulated trend
        trend = getattr(self, 'prediction_trend', 'NEUTRAL')
        
        if self.fear_greed_index < 25:
            if trend == 'RAPID_RECOVERY': score += 3.0 # Bottom reversal simulation
            else: score -= 1.0
        elif self.fear_greed_index > 75: score += 1.0 
        
        self.sentiment_score = score

    def get_short_bias(self) -> float:
        return -self.sentiment_score

    def get_allowed_sides(self) -> List[str]:
        # Same logic as strategy: One direction if strong alignment, otherwise flexible.
        if self.daily_bias == 'BULLISH' and self.regime == 'BULL':
            return ['LONG']
        elif self.daily_bias == 'BEARISH' and self.regime == 'BEAR':
            return ['SHORT']
        
        return ['LONG', 'SHORT']


class BacktestEngine:
    """
    Engine testing strategy within a specific date range.
    """
    def __init__(self, symbol: str, interval: str, start_date: str, end_date: str):
        self.symbol = symbol
        self.interval = interval
        self.start_str = start_date
        self.end_str = end_date
        
        self.adapter = BinanceAdapter()
        
        # --- NEW ADDED: HISTORICAL SENTIMENT ---
        # Now using class simulating real historical trend, not fake
        self.sentiment = HistoricalSentiment(self.adapter, start_date, end_date)
        # ------------------------------------------
        
        self.strategy = TradingStrategy(self.sentiment)
        self.db = DatabaseAdapter() 
        
        self.balance = settings.SIMULATED_BALANCE
        self.initial_balance = self.balance
        self.positions = []
        self.trade_history = []
        self.df = pd.DataFrame()

    def _str_to_timestamp(self, date_str: str) -> int:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.timestamp() * 1000)

    def fetch_historical_data(self):
        """Fetches data for the coin to be tested"""
        start_ts = self._str_to_timestamp(self.start_str)
        end_ts = self._str_to_timestamp(self.end_str)
        
        logger.info(f"⏳ Downloading Data for {self.symbol}: {self.start_str} -> {self.end_str}")
        
        all_klines = []
        current_start = start_ts
        
        while True:
            if current_start >= end_ts: break
            try:
                klines = self.adapter.client.futures_klines(
                    symbol=self.symbol, interval=self.interval, 
                    startTime=current_start, endTime=end_ts, limit=1000
                )
                if not klines: break
                all_klines.extend(klines)
                current_start = klines[-1][0] + 1
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Data fetch error: {e}")
                break
        
        if not all_klines:
            logger.error(f"❌ Data not found for {self.symbol}!")
            return

        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in cols: df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
            
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        self.df = df
        logger.info(f"✅ {self.symbol}: Total {len(df)} candles loaded.")

    def run(self):
        if self.df.empty: self.fetch_historical_data()
        if self.df.empty: return
            
        logger.info(f"🚀 Backtest Starting for {self.symbol}...")
        
        full_df = AdvancedTechnicalAnalysis.calculate_enhanced_indicators(self.df.copy())
        
        for i in range(50, len(full_df)):
            current_slice = full_df.iloc[:i+1].copy()
            current_row = current_slice.iloc[-1]
            current_price = current_row['close']
            current_time = current_row['timestamp']
            
            # --- CRITICAL UPDATE: Update Sentiment based on current time ---
            self.sentiment.update_for_time(current_time)
            # -----------------------------------------------------------------
            
            # 1. Open Position Management
            if self.positions:
                self._check_exit(current_slice, current_row, current_time)
                
            # 2. New Entry Check
            if not self.positions: 
                analysis = self.strategy.analyze_potential_entry(self.symbol, current_slice)
                
                if analysis:
                    self._execute_entry(analysis, current_price, current_time)

        # Cleanup
        if self.positions:
            last_price = full_df.iloc[-1]['close']
            last_time = full_df.iloc[-1]['timestamp']
            for pos in list(self.positions):
                self._close_position(pos, last_price, last_time, "End of Backtest")

        self._print_results()

    def _execute_entry(self, analysis: Dict, price: float, time: datetime):
        pos_type = analysis.get('type')
        score = analysis.get('score')
        detailed_scores = analysis.get('detailed_scores', {})
        
        margin = min(self.balance, settings.INVESTMENT_AMOUNT)
        qty = (margin * settings.LEVERAGE) / price
        tp_pct, sl_pct = self.strategy.calculate_targets(analysis)
        
        position = {
            'symbol': self.symbol,
            'position_type': pos_type,
            'buy_price': price,
            'quantity': qty,
            'amount': qty, 
            'buy_time': time,
            'leverage': settings.LEVERAGE,
            'margin_used': margin,
            'profit_target': tp_pct,
            'stop_loss': sl_pct,
            'score': score,
            'scores': detailed_scores,
            'highest_profit': 0.0,
            'max_profit_pct': 0.0, # New
            'max_loss_pct': 0.0    # New
        }
        self.positions.append(position)
        self.balance -= margin

    def _check_exit(self, df: pd.DataFrame, current_row: pd.Series, time: datetime):
        current_price = current_row['close']
        high_price = current_row['high']
        low_price = current_row['low']

        for pos in list(self.positions):
            # --- INTRA-CANDLE CHECK ---
            # Did TP/SL happen within candle movements without waiting for close?
            
            # --- Update Performance Statistics (with High/Low) ---
            buy_price = pos['buy_price']
            leverage = pos['leverage']
            
            if pos['position_type'] == 'LONG':
                roi_high = ((high_price - buy_price) / buy_price) * 100 * leverage
                roi_low = ((low_price - buy_price) / buy_price) * 100 * leverage
            else: # SHORT
                roi_high = ((buy_price - low_price) / buy_price) * 100 * leverage # Lower price = Higher profit
                roi_low = ((buy_price - high_price) / buy_price) * 100 * leverage # Higher price = Loss
            
            if roi_high > pos['max_profit_pct']: pos['max_profit_pct'] = roi_high
            if roi_low < pos['max_loss_pct']: pos['max_loss_pct'] = roi_low
            # -----------------------------------------------------------

            buy_price = pos['buy_price']
            sl_pct = pos['stop_loss']
            tp_pct = pos['profit_target']
            
            # Compare Hard Stop (Emergency Exit) Limit with Strategy Stop
            # Whichever is tighter (closer to 0) applies.
            # E.g.: sl_pct: -2.0, MAX_LOSS: 8.0 (-8.0). Max(-2, -8) = -2.
            # E.g.: sl_pct: -10.0, MAX_LOSS: 0.8 (-0.8). Max(-10, -0.8) = -0.8.
            effective_sl_pct = max(sl_pct, -settings.MAX_ALLOWED_LOSS_ON_EXIT)

            exit_price = None
            reason = None

            if pos['position_type'] == 'LONG':
                sl_price = buy_price * (1 + effective_sl_pct / 100)
                tp_price = buy_price * (1 + tp_pct / 100)
                
                # Did it hit Low first? (Stop Loss - Pessimistic Approach)
                if low_price <= sl_price:
                    exit_price = sl_price
                    reason = f"🛑 Stop Loss (Intra-Candle Low: {low_price:.4f})"
                elif high_price >= tp_price:
                    exit_price = tp_price
                    reason = f"🎯 Take Profit (Intra-Candle High: {high_price:.4f})"

            elif pos['position_type'] == 'SHORT':
                sl_price = buy_price * (1 - effective_sl_pct / 100) # Short SL is above
                tp_price = buy_price * (1 - tp_pct / 100)          # Short TP is below
                
                if high_price >= sl_price:
                    exit_price = sl_price
                    reason = f"🛑 Stop Loss (Intra-Candle High: {high_price:.4f})"
                elif low_price <= tp_price:
                    exit_price = tp_price
                    reason = f"🎯 Take Profit (Intra-Candle Low: {low_price:.4f})"

            if exit_price:
                self._close_position(pos, exit_price, time, reason)
                continue

            should_exit, reason = self.strategy.check_exit_conditions(pos, current_price, df)
            pos = self.strategy.update_trailing_stop(pos, current_price, df)
            if should_exit:
                self._close_position(pos, current_price, time, reason)

    def _close_position(self, pos: Dict, current_price: float, time: datetime, reason: str):
        entry_price = pos['buy_price']
        qty = pos['quantity']
        margin = pos['margin_used']
        
        if pos['position_type'] == 'SHORT':
            pnl = (entry_price - current_price) * qty
        else:
            pnl = (current_price - entry_price) * qty
            
        roi = (pnl / margin) * 100
        
        # Check closing ROI one last time
        if roi > pos['max_profit_pct']: pos['max_profit_pct'] = roi
        if roi < pos['max_loss_pct']: pos['max_loss_pct'] = roi

        self.balance += margin + pnl
        
        scores = pos.get('scores', {})
        trade_record = {
            'symbol': pos['symbol'],
            'type': pos['position_type'],
            'entry_time': pos['buy_time'].strftime('%Y-%m-%d %H:%M:%S'),
            'exit_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'entry_price': entry_price,
            'exit_price': current_price,
            'pnl': pnl,
            'roi': roi,
            'reason': reason,
            'score_total': scores.get('total', 0),
            'score_rsi': scores.get('score_rsi', 0),
            'score_macd': scores.get('score_macd', 0),
            'score_stoch': scores.get('score_stoch', 0),
            'score_williams': scores.get('score_williams', 0),
            'score_adx': scores.get('score_adx', 0),
            'score_ema': scores.get('score_ema', 0),
            'score_bb': scores.get('score_bb', 0),
            'score_sentiment': scores.get('score_sentiment', 0),
            'score_trend_health': scores.get('score_trend_health', 0),
            'score_volume': scores.get('score_volume', 0),
            'score_patterns': scores.get('score_patterns', 0),
            'score_bonus': scores.get('score_bonus', 0),
            'max_profit_pct': pos['max_profit_pct'],
            'max_loss_pct': pos['max_loss_pct']
        }
        self.db.save_trade(trade_record)
        self.trade_history.append(trade_record)
        self.positions.remove(pos)

    def _print_results(self):
        print("\n" + "="*40)
        print(f"📊 BACKTEST RESULT: {self.symbol}")
        print(f"Period: {self.start_str} -> {self.end_str}")
        print("="*40)
        print(f"Start     : {self.initial_balance:.2f} USDT")
        print(f"End       : {self.balance:.2f} USDT")
        print(f"Net PnL   : {self.balance - self.initial_balance:.2f} USDT")
        
        total_trades = len(self.trade_history)
        if total_trades > 0:
            wins = len([t for t in self.trade_history if t['pnl'] > 0])
            win_rate = (wins / total_trades) * 100
            print(f"Trade Count : {total_trades}")
            print(f"Win Rate    : %{win_rate:.2f}")
            print("-" * 40)
            print("Last 5 Trades:")
            for t in self.trade_history[-5:]:
                print(f"{t['type']} | {t['entry_time']} | PnL: {t['pnl']:.2f}$ | {t['reason']}")
        else:
            print("No trades opened.")
        print("="*40)