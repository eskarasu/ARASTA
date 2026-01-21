import sys
import os

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)
# ---------------------

import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Optional

# --- PROJECT MODULES ---
from src.config.settings import settings
from src.infrastructure.logger import logger
from src.infrastructure.binance_api import BinanceAdapter
from src.infrastructure.database_adapter import LiveDatabaseAdapter
from src.infrastructure.analysis import AdvancedTechnicalAnalysis
from src.application.services import EnhancedRiskManager
from src.application.strategies import TradingStrategy, MarketSentiment

class ArastaBot:
    """
    Main Orchestration Class of the Bot.
    Updated with detailed statistics tracking and hourly reporting features.
    """

    def __init__(self):
        # 1. Configuration Validation
        settings.validate()
        
        # 2. Initialize Infrastructure Layer
        self.exchange = BinanceAdapter()
        
        # 3. Initialize Application Layer
        self.risk_manager = EnhancedRiskManager()
        self.sentiment_analyzer = MarketSentiment()
        self.strategy = TradingStrategy(self.sentiment_analyzer)
        self.live_db = LiveDatabaseAdapter()
        
        # 4. Bot State Variables
        self.portfolio: Dict[str, Dict] = {}  # Open positions
        self.volume_leaders = []              # Coin list to watch
        self.recently_sold = {}               # For cooldown tracking
        self.session_start = datetime.now()
        
        # 5. DETAILED STATISTICS TRACKING
        self.stats = {
            'total_pnl_usdt': 0.0,
            'peak_balance': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_short': 0,
            'winning_short': 0,
            'losing_short': 0,
            'total_long': 0,
            'winning_long': 0,
            'losing_long': 0
        }
        
        self.last_hourly_report = time.time()

    def initialize(self):
        """Preparations before bot starts"""
        logger.info("🤖 ARASTA v2.0 Starting...")
        logger.info("🤖 Adaptive Regime And Sentiment Trading Agent Starting...")
        logger.info(f"Mode: {'🔥 LIVE TRADING' if settings.LIVE_TRADING else '🧪 SIMULATION'}")
        logger.info(f"Market: {'FUTURES' if settings.USE_FUTURES else 'SPOT'}")
        
        if self.exchange.check_connection():
            logger.info("✅ Exchange connection successful.")
        else:
            logger.error("❌ Exchange connection failed! Exiting.")
            exit(1)

        # Get initial balance
        balance = self.exchange.get_balance()
        self.risk_manager.current_balance = balance if settings.LIVE_TRADING else settings.SIMULATED_BALANCE
        self.stats['peak_balance'] = self.risk_manager.current_balance
        
        logger.info(f"💰 Initial Balance: {self.risk_manager.current_balance:.2f} USDT")

    def _update_stats(self, pnl_amount: float, is_win: bool, position_type: str):
        """Updates statistics after trade"""
        self.stats['total_trades'] += 1
        self.stats['total_pnl_usdt'] += pnl_amount
        
        if is_win:
            self.stats['winning_trades'] += 1
        else:
            self.stats['losing_trades'] += 1
            
        if position_type == 'SHORT':
            self.stats['total_short'] += 1
            if is_win: self.stats['winning_short'] += 1
            else: self.stats['losing_short'] += 1
        else:
            self.stats['total_long'] += 1
            if is_win: self.stats['winning_long'] += 1
            else: self.stats['losing_long'] += 1
            
        current_balance = self.risk_manager.current_balance
        if current_balance > self.stats['peak_balance']:
            self.stats['peak_balance'] = current_balance
        
        if self.stats['peak_balance'] > 0:
            drawdown = (self.stats['peak_balance'] - current_balance) / self.stats['peak_balance'] * 100
            if drawdown > self.stats['max_drawdown']:
                self.stats['max_drawdown'] = drawdown

    def _log_detailed_report(self, title="📊 HOURLY PERFORMANCE REPORT"):
        """Prints very detailed status table to console"""
        s = self.stats
        wr_total = (s['winning_trades'] / s['total_trades'] * 100) if s['total_trades'] > 0 else 0
        wr_short = (s['winning_short'] / s['total_short'] * 100) if s['total_short'] > 0 else 0
        wr_long = (s['winning_long'] / s['total_long'] * 100) if s['total_long'] > 0 else 0
        
        uptime = datetime.now() - self.session_start
        uptime_str = str(uptime).split('.')[0]

        report = f"""
    {'='*60}
    {title:^60}
    {'='*60}
    ⏱️  Uptime         : {uptime_str}
    💰 Current Balance: {self.risk_manager.current_balance:.2f} USDT
    💹 Total Profit   : {s['total_pnl_usdt']:+.2f} USDT
     Max Drawdown   : %{s['max_drawdown']:.2f}
    ────────────────────────────────────────────────────────────
    GENERAL STATISTICS:
      🔄 Total Trades : {s['total_trades']}
      🏆 Win Rate     : %{wr_total:.1f} ({s['winning_trades']} Wins / {s['losing_trades']} Losses)
    ────────────────────────────────────────────────────────────
    SHORT TRADES:
      📉 Total        : {s['total_short']}
      ✅ Successful   : {s['winning_short']}
      ❌ Failed       : {s['losing_short']}
       Short WR     : %{wr_short:.1f}
    ────────────────────────────────────────────────────────────
    LONG TRADES:
      📈 Total        : {s['total_long']}
      ✅ Successful   : {s['winning_long']}
      ❌ Failed       : {s['losing_long']}
      📊 Long WR      : %{wr_long:.1f}
    {'='*60}
        """
        logger.info(report)

    def _check_hourly_report(self):
        """Hourly report check"""
        current_time = time.time()
        if current_time - self.last_hourly_report >= 3600:
            self._log_detailed_report()
            self.last_hourly_report = current_time

    def _update_market_data(self):
        try:
            self.volume_leaders = self.exchange.get_top_volatility_symbols(limit=85)
            logger.info(f"📊 Watchlist Updated: {len(self.volume_leaders)} symbols (Volatility Based)")
            self.sentiment_analyzer.update(self.exchange.client)
        except Exception as e:
            logger.error(f"Market data update error: {e}")

    def _get_dataframe(self, symbol: str) -> pd.DataFrame:
        return self.exchange.get_market_data(symbol, settings.LOOKUP_INTERVAL)

    def _scan_opportunities(self):
        stop_trading, reason = self.risk_manager.should_stop_trading()
        if stop_trading:
            logger.warning(f"⛔ RISK LIMIT: Trading stopped. Reason: {reason}")
            return

        if len(self.portfolio) >= settings.MAX_HOLDINGS:
            return

        logger.info(f"🔍 Scanning Market ({len(self.volume_leaders)} symbols)...")
        
        for symbol in self.volume_leaders:
            if symbol in self.portfolio: continue
            
            if symbol in self.recently_sold:
                if (time.time() - self.recently_sold[symbol]) < settings.RECENTLY_SOLD_COOLDOWN:
                    continue
                else:
                    del self.recently_sold[symbol]

            df = self._get_dataframe(symbol)
            if df.empty: continue
            
            opportunity = self.strategy.analyze_potential_entry(symbol, df)
            
            if opportunity:
                self._execute_entry(opportunity)
                
                if len(self.portfolio) >= settings.MAX_HOLDINGS:
                    break

    def _execute_entry(self, opportunity: Dict):
        symbol = opportunity['symbol']
        signal_type = opportunity.get('type', 'LONG')
        price = opportunity['current_price']
        scores = opportunity.get('detailed_scores', {})
        volatility = opportunity['volatility']
        
        if signal_type == 'SHORT':
            allowed, msg = self.risk_manager.should_allow_short()
            if not allowed:
                logger.warning(f"Short trade blocked: {msg}")
                return
            usdt_amount = self.risk_manager.calculate_short_position_size(self.risk_manager.current_balance, volatility)
        else:
            usdt_amount = self.risk_manager.calculate_position_size(symbol, self.risk_manager.current_balance, volatility)

        tp_pct, sl_pct = self.strategy.calculate_targets(opportunity)
        leverage = settings.LEVERAGE if settings.USE_FUTURES else 1
        quantity = (usdt_amount * leverage) / price

        self.exchange.set_leverage(symbol, leverage)
        self.exchange.set_margin_type(symbol, settings.MARGIN_TYPE)
        
        order_result = self.exchange.place_market_order(symbol, 'SELL' if signal_type == 'SHORT' else 'BUY', quantity)
        
        if order_result:
            executed_price = float(order_result.get('avgPrice', price))
            if executed_price <= 0: executed_price = price
            
            # --- FIX: If executedQty is 0, use calculated quantity ---
            final_qty_api = float(order_result.get('executedQty', 0.0))
            if final_qty_api > 0:
                final_qty = final_qty_api
            else:
                # If 0 returned from Simulation or API, use our formatted amount (might come as string in executedQty)
                # or use original quantity.
                final_qty = float(order_result.get('executedQty') or quantity)
                if final_qty == 0: final_qty = quantity # Last resort
            # ---------------------------------------------------------------
            
            # --- DB RECORD (OPENING) ---
            trade_id = -1
            try:
                open_record = {
                    'symbol': symbol,
                    'type': signal_type,
                    'entry_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'entry_price': executed_price,
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
                    'score_bonus': scores.get('score_bonus', 0)
                }
                trade_id = self.live_db.log_open_trade(open_record)
            except Exception as e:
                logger.error(f"DB Opening Record Error: {e}")
            # -------------------------

            self.portfolio[symbol] = {
                'symbol': symbol,
                'position_type': signal_type,
                'buy_price': executed_price,
                'amount': final_qty,
                'buy_time': datetime.now(),
                'profit_target': tp_pct,
                'stop_loss': sl_pct,
                'initial_investment': usdt_amount,
                'leverage': leverage, 
                'scores': scores,
                'db_id': trade_id,  # Storing ID to update on exit
                'max_profit_pct': 0.0, # New: Highest profit ratio
                'max_loss_pct': 0.0    # New: Highest loss ratio
            }
            
            self.risk_manager.record_trade(0, is_short=(signal_type=='SHORT'))
            logger.info(f"✅ POSITION OPENED: {symbol} | Side: {signal_type} | Score: {opportunity['score']:.1f} | Price: {executed_price:.4f}")
            
            if not settings.LIVE_TRADING:
                self.risk_manager.current_balance -= usdt_amount
        else:
            logger.error(f"Order failed: {symbol}")

    def _monitor_positions_loop(self):
        while True:
            try:
                self._check_hourly_report()

                if not self.portfolio:
                    time.sleep(10)
                    continue

                for symbol in list(self.portfolio.keys()):
                    self._check_single_position(symbol)
                
                time.sleep(settings.PRICE_CHECK_INTERVAL)
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(10)

    def _check_single_position(self, symbol: str):
        if symbol not in self.portfolio: return
        
        pos = self.portfolio[symbol]
        entry_price = pos['buy_price']

        if entry_price <= 0:
            logger.error(f"❌ invalid entry price (0.0) for {symbol}. Closing position.")
            self._execute_exit(symbol, self.exchange.get_price(symbol), "Invalid Entry Price", 0.0)
            return

        current_price = self.exchange.get_price(symbol)
        if not current_price: return
        
        df = self._get_dataframe(symbol)
        should_close, reason = self.strategy.check_exit_conditions(pos, current_price, df)
        self.portfolio[symbol] = self.strategy.update_trailing_stop(pos, current_price, df)
        
        if pos['position_type'] == 'SHORT':
            pnl_pct = ((entry_price - current_price) / entry_price) * 100 * pos['leverage']
        else:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100 * pos['leverage']
            
        # --- Update Performance Statistics ---
        if pnl_pct > pos['max_profit_pct']: pos['max_profit_pct'] = pnl_pct
        if pnl_pct < pos['max_loss_pct']: pos['max_loss_pct'] = pnl_pct
        
        # Reflect to DB (For Live Monitoring)
        if pos.get('db_id', -1) != -1:
            self.live_db.update_trade_performance(pos['db_id'], pos['max_profit_pct'], pos['max_loss_pct'])
        # --------------------------------------------

        logger.info(f"👀 Watching: {symbol:10} | PnL: %{pnl_pct:>6.2f} | Max: %{pos['max_profit_pct']:.1f} | Min: %{pos['max_loss_pct']:.1f} | Status: {reason if should_close else 'Waiting'}")

        if should_close:
            self._execute_exit(symbol, current_price, reason, pnl_pct)

    def _execute_exit(self, symbol: str, current_price: float, reason: str, pnl_pct: float):
        if symbol not in self.portfolio: return
        
        pos = self.portfolio[symbol]
        qty = pos['amount']

        # If amount is 0 delete and exit
        if qty <= 0:
            logger.error(f"❌ {symbol} cannot close, amount 0! Removing from portfolio.")
            del self.portfolio[symbol]
            return

        side = 'BUY' if pos['position_type'] == 'SHORT' else 'SELL'
        
        logger.info(f"🚨 CLOSING POSITION: {symbol} | Amount: {qty} | Reason: {reason}")
        order_result = self.exchange.place_market_order(symbol, side, qty, reduce_only=True)
        
        if order_result:
            realized_pnl_usd = pos['initial_investment'] * (pnl_pct / 100) 
            
            # Update one last time (With closing price)
            if pnl_pct > pos['max_profit_pct']: pos['max_profit_pct'] = pnl_pct
            if pnl_pct < pos['max_loss_pct']: pos['max_loss_pct'] = pnl_pct

            self.risk_manager.record_trade(pnl_pct, is_short=(pos['position_type']=='SHORT'))
            
            is_win = realized_pnl_usd > 0
            self._update_stats(realized_pnl_usd, is_win, pos['position_type'])
            
            if not settings.LIVE_TRADING:
                self.risk_manager.current_balance += (pos['initial_investment'] + realized_pnl_usd)
            else:
                time.sleep(2)
                self.risk_manager.current_balance = self.exchange.get_balance()
            
            self.recently_sold[symbol] = time.time()
            del self.portfolio[symbol]
            
            # --- DB RECORD ---
            try:
                trade_id = pos.get('db_id', -1)
                
                # If ID exists update, else (if trade from old version) insert new record
                if trade_id != -1:
                    exit_data = {
                        'exit_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'exit_price': current_price,
                        'pnl': realized_pnl_usd,
                        'roi': pnl_pct,
                        'reason': reason,
                        'max_profit_pct': pos['max_profit_pct'],
                        'max_loss_pct': pos['max_loss_pct']
                    }
                    self.live_db.update_trade_exit(trade_id, exit_data)
                else:
                    # Fallback: If no ID save old way (save_trade function still exists)
                    # But here we log instead of calling save_trade, because we didn't change save_trade
                    logger.warning(f"DB ID not found, trade could not be updated: {symbol}")
                    
            except Exception as e:
                logger.error(f"DB Save Error: {e}")
            # ----------------
            
            logger.info(f"💰 TRADE RESULT: {symbol} | Net PnL: {realized_pnl_usd:+.2f} USDT | Balance: {self.risk_manager.current_balance:.2f}")
            self._log_detailed_report("📊 POST-TRADE REPORT")

    def run(self):
        self.initialize()
        monitor_thread = threading.Thread(target=self._monitor_positions_loop, daemon=True)
        monitor_thread.start()
        logger.info("🚀 Loop Started. You can stop with Ctrl+C.")
        
        try:
            while True:
                self._update_market_data()
                self._scan_opportunities()
                logger.info(f"💤 Sleep Mode ({settings.PRICE_CHECK_INTERVAL}s)... Open Position: {len(self.portfolio)}")
                time.sleep(settings.PRICE_CHECK_INTERVAL)
        except KeyboardInterrupt:
            logger.warning("🛑 Bot stopped by user.")
            self.print_summary()
        except Exception as e:
            logger.exception(f"🔥 CRITICAL ERROR: {e}")

    def print_summary(self):
        self._log_detailed_report("🏁 CLOSING REPORT")

if __name__ == '__main__':
    bot = ArastaBot()
    bot.run()