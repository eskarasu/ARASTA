import time
import numpy as np
import pandas as pd
from math import floor
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from binance.client import Client
from binance.exceptions import BinanceAPIException
from requests.exceptions import RequestException

from src.config.settings import settings
from src.infrastructure.logger import logger

class BinanceAdapter:
    """
    Binance API Adapter.
    Combines precision management and 'Old Bot Aggressive Scanning' logic.
    Supports both Live and Simulation modes.
    """

    def __init__(self):
        try:
            # According to variable name in settings file (API_KEY)
            self.client = Client(settings.API_KEY, settings.API_SECRET)
            self.use_futures = settings.USE_FUTURES
            
            # Simulation Data
            self.simulation_positions = []
            self.simulated_balance = settings.SIMULATED_BALANCE
            
            # Cache symbol info (For performance)
            self.symbol_info_cache = {}
            
            logger.info(f"Binance Adapter Initialized. Mode: {'FUTURES' if self.use_futures else 'SPOT'}")
        except AttributeError:
            logger.error("ERROR: 'API_KEY' not found in settings.py.")
            raise
        except Exception as e:
            logger.error(f"Binance Client could not be initialized: {e}")
            raise

    def check_connection(self) -> bool:
        """Tests API connection"""
        try:
            if self.use_futures:
                self.client.futures_account()
            else:
                self.client.get_account()
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def get_balance(self, asset: str = 'USDT') -> float:
        """Fetches current balance (Simulation supported)"""
        if not settings.LIVE_TRADING:
            return self.simulated_balance

        try:
            if self.use_futures:
                balances = self.client.futures_account_balance()
                target = next((b for b in balances if b['asset'] == asset), None)
                if target:
                    return float(target.get('withdrawAvailable', target.get('balance', 0.0)))
            else:
                balance = self.client.get_asset_balance(asset=asset)
                if balance:
                    return float(balance['free'])
            return 0.0
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0.0

    def get_price(self, symbol: str) -> Optional[float]:
        """Fetches current price of the symbol"""
        try:
            if self.use_futures:
                try:
                    ticker = self.client.futures_symbol_ticker(symbol=symbol)
                    return float(ticker['price'])
                except:
                    ticker = self.client.futures_mark_price(symbol=symbol)
                    return float(ticker.get('markPrice', 0))
            else:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                return float(ticker['price'])
        except Exception as e:
            logger.warning(f"Price could not be fetched ({symbol}): {e}")
            return None

    def get_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Fetches historical candle data (Klines)"""
        try:
            if self.use_futures:
                klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)
            else:
                klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)

            if not klines:
                return pd.DataFrame()

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Numeric conversion
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df

        except Exception as e:
            logger.error(f"Kline data could not be fetched ({symbol}): {e}")
            return pd.DataFrame()

    def get_market_data(self, symbol: str, interval: str) -> pd.DataFrame:
        """Alias for get_klines (for main.py compatibility)"""
        return self.get_klines(symbol, interval, settings.ANALYSIS_PERIOD)

    def get_top_volatility_symbols(self, limit: int = 85) -> List[str]:
        """
        Finds 'Exploding' coins using Volume + Price Change formula.
        """
        try:
            if self.use_futures:
                tickers = self.client.futures_ticker()
            else:
                tickers = self.client.get_ticker()

            scored_pairs = []
            
            for t in tickers:
                symbol = t['symbol']
                
                if not symbol.endswith('USDT'): continue
                if symbol in settings.SYMBOL_BLACKLIST: continue
                if "UPUSDT" in symbol or "DOWNUSDT" in symbol: continue 
                
                try:
                    # --- Legacy Volatility & Volume Formula ---
                    quote_vol = float(t['quoteVolume'])
                    price_change = abs(float(t['priceChangePercent']))
                    count = float(t['count'])
                    
                    # Scoring: Volume + Volatility + Trade Count
                    score = (quote_vol * 0.6) + (price_change * quote_vol * 0.3) + (count * 0.1)
                    
                    scored_pairs.append({
                        'symbol': symbol,
                        'score': score
                    })
                except:
                    continue

            # Sort by score (Highest score at top)
            scored_pairs.sort(key=lambda x: x['score'], reverse=True)
            
            # Get only symbol names
            top_symbols = [x['symbol'] for x in scored_pairs[:limit]]
            
            # Add favorite symbols to top
            for sym in reversed(sorted(list(settings.PREFERRED_SYMBOLS))):
                if sym in top_symbols:
                    top_symbols.remove(sym)
                    top_symbols.insert(0, sym)
                elif sym not in settings.SYMBOL_BLACKLIST:
                    top_symbols.insert(0, sym)
            
            return list(dict.fromkeys(top_symbols))[:limit]

        except Exception as e:
            logger.error(f"Symbol scanning error: {e}")
            return []
            
    # Alias definitions
    get_top_volume_symbols = get_top_volatility_symbols

    def get_open_positions(self) -> List[Dict]:
        """Fetches open positions"""
        if not settings.LIVE_TRADING:
            return self.simulation_positions

        positions = []
        try:
            if self.use_futures:
                acc = self.client.futures_account()
                for pos in acc['positions']:
                    amt = float(pos['positionAmt'])
                    if amt != 0:
                        positions.append({
                            'symbol': pos['symbol'],
                            'amount': abs(amt),
                            'entry_price': float(pos['entryPrice']),
                            'pnl': float(pos['unrealizedProfit']),
                            'side': 'LONG' if amt > 0 else 'SHORT',
                            'buy_price': float(pos['entryPrice']),
                            'quantity': abs(amt),
                            'position_type': 'LONG' if amt > 0 else 'SHORT',
                            'buy_time': datetime.now(),
                            'profit_target': 999, # Managed by strategy file
                            'stop_loss': -999
                        })
            return positions
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []

    def get_symbol_info(self, symbol: str) -> Dict:
        """Fetches symbol filter info (Cache supported)"""
        if symbol in self.symbol_info_cache:
            return self.symbol_info_cache[symbol]

        try:
            if self.use_futures:
                exchange_info = self.client.futures_exchange_info()
            else:
                exchange_info = self.client.get_exchange_info()

            for s in exchange_info['symbols']:
                if s['symbol'] == symbol:
                    self.symbol_info_cache[symbol] = s
                    return s
            return {}
        except Exception as e:
            logger.error(f"Symbol info could not be fetched ({symbol}): {e}")
            return {}

    def _format_quantity(self, symbol: str, quantity: float) -> str:
        """
        Formats quantity according to Binance rules (LOT_SIZE).
        FIX: Rounding errors fixed by adding Epsilon.
        """
        try:
            info = self.get_symbol_info(symbol)
            if not info: return str(int(quantity))

            lot_filter = next((f for f in info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            if not lot_filter: return str(int(quantity))

            step_size = float(lot_filter['stepSize'])
            min_qty = float(lot_filter['minQty'])
            
            if step_size <= 0: return str(int(quantity))
            
            # If quantity is less than min, return 0 (cannot trade)
            if quantity < min_qty:
                return "0"

            # Calculate Precision
            precision = int(round(-np.log10(step_size), 0))
            
            # Epsilon: Add very small number to prevent floating point errors
            epsilon = 1e-9
            
            # Rounding Logic:
            quantity_adjusted = floor((quantity + epsilon) / step_size) * step_size
            
            return "{:0.{precision}f}".format(quantity_adjusted, precision=precision)
            
        except Exception as e:
            logger.warning(f"⚠️ Quantity formatting error ({symbol}): {e}")
            return str(int(quantity))

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Sets leverage"""
        if not self.use_futures: return True
        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            return True
        except Exception: return False

    # --- NEW ADDED METHOD ---
    def set_margin_type(self, symbol: str, margin_type: str) -> bool:
        """Sets margin type (ISOLATED / CROSSED)"""
        if not self.use_futures: return True
        try:
            self.client.futures_change_margin_type(symbol=symbol, marginType=margin_type)
            return True
        except BinanceAPIException as e:
            # Code -4046: "No need to change margin type" (Already in that mode)
            if e.code == -4046: return True
            return False
        except Exception: return False
    # --------------------------

    def place_market_order(self, symbol: str, side: str, quantity: float, reduce_only: bool = False) -> Optional[Dict]:
        """Places market order"""
        
        # 1. Calculate Quantity
        formatted_qty = self._format_quantity(symbol, quantity)
        qty_float = float(formatted_qty)
        
        # 2. Zero Quantity Check (ERROR PREVENTED HERE)
        if qty_float <= 0:
            logger.warning(f"❌ Calculated quantity for {symbol} is 0! (Raw: {quantity:.6f}). Order cancelled.")
            return None
        
        logger.info(f"SENDING ORDER: {symbol} {side} {formatted_qty}")
        
        # SIMULATION
        if not settings.LIVE_TRADING:
            logger.info("🧪 Simulation Mode: Order not sent to exchange.")
            price = self.get_price(symbol) or 0.0
            
            # Deduct from balance (For calculation only, position tracking in main.py)
            cost = qty_float * price
            if settings.USE_FUTURES: cost = cost / settings.LEVERAGE
            self.simulated_balance -= cost

            return {
                'symbol': symbol,
                'orderId': 'SIM_ORDER',
                'executedQty': formatted_qty,
                'avgPrice': price,
                'side': side,
                'status': 'FILLED'
            }

        # LIVE
        try:
            if self.use_futures:
                self.set_leverage(symbol, settings.LEVERAGE)
                # main.py sets Margin Type but keeping here as backup
                # self.set_margin_type(symbol, settings.MARGIN_TYPE)
                
            if self.use_futures:
                return self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=formatted_qty,
                    reduceOnly=reduce_only
                )
            else:
                if side == 'BUY':
                    return self.client.order_market_buy(symbol=symbol, quantity=formatted_qty)
                else:
                    return self.client.order_market_sell(symbol=symbol, quantity=formatted_qty)
                    
        except BinanceAPIException as e:
            logger.error(f"❌ Exchange Error (API): {e.message} Code: {e.code}")
            return None
        except Exception as e:
            logger.error(f"❌ Exchange Error (General): {e}")
            return None

    def calculate_pnl(self, position: Dict, current_price: float) -> Dict:
        """Calculates instant PnL and ROI"""
        entry_price = position['buy_price']
        qty = position['quantity']
        pos_type = position.get('position_type', 'LONG')
        
        if pos_type == 'SHORT':
            pnl_amount = (entry_price - current_price) * qty
            roi = ((entry_price - current_price) / entry_price) * 100 * settings.LEVERAGE
        else:
            pnl_amount = (current_price - entry_price) * qty
            roi = ((current_price - entry_price) / entry_price) * 100 * settings.LEVERAGE
            
        return {'pnl': pnl_amount, 'roi': roi}