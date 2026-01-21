import time
import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Optional, Any, List
from collections import deque

# Settings and Logger (Adjust according to your project paths)
from src.config.settings import settings
from src.infrastructure.logger import logger
from src.infrastructure.analysis import AdvancedTechnicalAnalysis
from src.infrastructure.sentiment_provider import SentimentProvider

class MarketSentiment:
    """
    Smart Module determining Market Regime (Trend Direction).
    Looks at EMA 50/200 relationship on BTC 4H chart.
    """
    def __init__(self):
        self.regime = 'NEUTRAL' # BULL, BEAR, NEUTRAL
        self.btc_trend = 'NEUTRAL' # BULLISH, BEARISH
        self.daily_bias = 'NEUTRAL' # NEW: Daily Bias (Macro Bias)
        self.fear_greed_index = 50
        self.sentiment_score = 0.0 # <-- THIS VARIABLE WAS MISSING, ADDED
        self.last_update = 0
        # New Provider
        self.provider = SentimentProvider()
        self.prediction = {}

    def update(self, client):
        """Updates market direction (Regime)"""
        # Updating every 1 hour is sufficient (Trend doesn't change often)
        if time.time() - self.last_update < 3600: 
            return

        try:
            # 1. Fetch BTC 4-Hour Data (For more reliable trend)
            klines = client.get_klines(symbol='BTCUSDT', interval='4h', limit=210)
            if not klines: return
            
            closes = np.array([float(k[4]) for k in klines])
            
            # 2. Calculate EMA (Manual or with TA-Lib)
            import talib
            ema_50 = talib.EMA(closes, timeperiod=50)[-1]
            ema_200 = talib.EMA(closes, timeperiod=200)[-1]
            rsi_14 = talib.RSI(closes, timeperiod=14)[-1] # For Synthetic F&G
            current_price = closes[-1]
            
            # 3. REGIME DETERMINATION (Decision Mechanism)
            if current_price > ema_50 and ema_50 > ema_200:
                self.regime = 'BULL' # Uptrend
                self.btc_trend = 'BULLISH'
                logger.info(f"🌊 MARKET REGIME: 🟢 BULL (Price: {current_price} > EMA50 > EMA200)")
                
            elif current_price < ema_50 and ema_50 < ema_200:
                self.regime = 'BEAR' # Downtrend
                self.btc_trend = 'BEARISH'
                logger.info(f"🌊 MARKET REGIME: 🔴 BEAR (Price: {current_price} < EMA50 < EMA200)")
                
            else:
                self.regime = 'NEUTRAL' # Sideways/Undecided
                self.btc_trend = 'NEUTRAL'
                logger.info(f"🌊 MARKET REGIME: 🦀 SIDEWAYS (Mixed Signal)")
            
            # --- NEW: DAILY TREND (MACRO BIAS) ANALYSIS ---
            try:
                # Fetch daily candles (Last 60 days)
                klines_1d = client.get_klines(symbol='BTCUSDT', interval='1d', limit=60)
                if klines_1d:
                    closes_1d = np.array([float(k[4]) for k in klines_1d])
                    ema_50_d = talib.EMA(closes_1d, timeperiod=50)[-1]
                    price_d = closes_1d[-1]
                    
                    if price_d > ema_50_d:
                        self.daily_bias = 'BULLISH'
                    else:
                        self.daily_bias = 'BEARISH'
                    logger.info(f"📅 DAILY BIAS: {self.daily_bias} (Price: {price_d:.0f} vs EMA50: {ema_50_d:.0f})")
            except Exception as e:
                logger.warning(f"Daily trend analysis error: {e}")
            # -----------------------------------------------

            # Fear & Greed update... (Via Provider)
            try:
                # Get prediction and current data from Provider
                self.prediction = self.provider.predict_next_move()
                self.fear_greed_index = self.prediction.get('current_value', 50)
                logger.info(f"🧠 SENTIMENT PREDICTION: {self.prediction.get('trend')} (Target: {self.prediction.get('predicted_value'):.1f})")
            except:
                self.fear_greed_index = 50

            # --- ADDED PART: Calculate Sentiment Score ---
            self.calculate_score()
            # ----------------------------------------------

            self.last_update = time.time()
            
        except Exception as e:
            logger.warning(f"Trend analysis error: {e}")

    # --- ADDED METHOD: Sentiment Score Calculation ---
    def calculate_score(self):
        score = 0.0
        # Score based on BTC Trend
        if self.regime == 'BULL': score += 2.0
        elif self.regime == 'BEAR': score -= 2.0
        
        # Fear & Greed Score
        # NOW SMARTER: We look at trend, not just value.
        
        trend = self.prediction.get('trend', 'NEUTRAL')
        
        if self.fear_greed_index < 20:
            if trend == 'RAPID_RECOVERY': score += 3.0 # "Buy when there's blood" (Bottom reversal)
            else: score -= 1.0 # Still falling, wait
            
        elif self.fear_greed_index > 75:
            if trend == 'RAPID_FEAR': score -= 3.0 # Reversal from top (Sell signal)
            else: score += 1.0 # Trend continues strong
            
        elif trend == 'RAPID_RECOVERY': score += 1.5 # General recovery
        
        self.sentiment_score = score
    # ------------------------------------------------

    def get_short_bias(self) -> float:
        """Market suitability score for Short trades"""
        # For Short, 'Bad' market is good (High score)
        # So if sentiment_score is negative, short bias should be positive.
        return -self.sentiment_score 

    def get_allowed_sides(self) -> List[str]:
        """Tells which direction to trade"""
        
        # 1. STRONG TREND ALIGNMENT (Daily + 4H same direction)
        # If both Daily and 4H are Bull, open ONLY LONG.
        if self.daily_bias == 'BULLISH' and self.regime == 'BULL':
            return ['LONG']
            
        # If both Daily and 4H are Bear, open ONLY SHORT.
        if self.daily_bias == 'BEARISH' and self.regime == 'BEAR':
            return ['SHORT']
            
        # 2. MIXED SITUATIONS (Be Flexible)
        if self.regime == 'BULL':
            return ['LONG', 'SHORT'] # If Daily Bearish, Short opportunity possible (Pullback)
        elif self.regime == 'BEAR':
            return ['LONG', 'SHORT'] # If Daily Bullish, Long opportunity possible (Dip buy)
        else:
            return ['LONG', 'SHORT']

class TradingStrategy:
    """Main Trading Strategy"""
    
    def __init__(self, market_sentiment: MarketSentiment):
        self.sentiment = market_sentiment

    def calculate_trend_health(self, df: pd.DataFrame) -> Tuple[float, List[str]]:
        """
        Exact same logic as in spaghetti code.
        """
        if df.empty or len(df) < 20:
            return 0.0, []

        reasons = []
        score_modifier = 0.0
        
        try:
            closes = df['close'].values
            if 'RSI' not in df.columns: return 0.0, []
            rsi = df['RSI'].values
            volumes = df['volume'].values
            last = df.iloc[-1]
            
            # Parameters (From Env or default)
            bb_thresh = settings.SMART_ENTRY_BB_THRESHOLD
            rsi_thresh = settings.SMART_ENTRY_RSI_THRESHOLD
            mom_thresh = settings.FRESH_SIGNAL_MIN_MOMENTUM
            vol_ratio_thresh = settings.FRESH_SIGNAL_VOLUME_RATIO
            
            # --- 1. HARD FILTERS (Peak/Risk Control) ---
            # BB Ceiling
            if 'BB_Upper' in last and 'BB_Lower' in last:
                bb_range = last['BB_Upper'] - last['BB_Lower']
                if bb_range > 0:
                    bb_pos = (last['close'] - last['BB_Lower']) / bb_range
                    if bb_pos > bb_thresh:
                        return -100.0, [f"⛔ Price at BB Ceiling ({bb_pos:.2f})"]

            # RSI Extremely Bloated
            if last['RSI'] > rsi_thresh:
                return -100.0, [f"⛔ RSI Extremely Bloated ({last['RSI']:.1f})"]

            # Bearish Divergence
            if len(closes) > 5:
                price_change_5 = (closes[-1] - closes[-5]) / closes[-5]
                rsi_change_5 = rsi[-1] - rsi[-5]
                if price_change_5 > 0.02 and rsi_change_5 < -5:
                    return -100.0, ["⛔ Bearish Divergence"]

            # --- 2. SOFT FILTERS (Momentum/Freshness Score) ---
            
            # A. Momentum
            if len(closes) > 10:
                price_momentum = (closes[-1] - closes[-10]) / closes[-10] * 100
                mom_recent = abs(closes[-1] - closes[-3]) if len(closes) >= 3 else 0
                mom_prev = abs(closes[-3] - closes[-6]) if len(closes) >= 6 else 0

                if price_momentum > mom_thresh:
                    if mom_recent > mom_prev:
                        score_modifier += 2.0
                        reasons.append(f"🚀 Momentum Accelerating (+{price_momentum:.1f}%)")
                    elif mom_recent < mom_prev * 0.5:
                        score_modifier -= 3.0
                        reasons.append("🐌 Momentum Slowing (Fatigue)")
                    else:
                        score_modifier += 1.0
                elif price_momentum < 0:
                    score_modifier -= 2.0
                    reasons.append("🐌 Price Momentum Negative")

            # B. RSI Slope
            if len(rsi) > 5:
                rsi_trend = (rsi[-1] - rsi[-5]) / 5
                if rsi_trend > 0.5:
                    score_modifier += 1.5
                    reasons.append("📈 RSI in Uptrend")
                elif rsi_trend < -0.5:
                    score_modifier -= 2.0 
                    reasons.append("📉 RSI in Downtrend")

            # C. Volume Increase
            if len(volumes) > 20:
                recent_vol = np.mean(volumes[-5:])
                avg_vol = np.mean(volumes[-20:-5])
                vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 0
                
                if vol_ratio > vol_ratio_thresh:
                    score_modifier += 1.5
                    reasons.append(f"📊 Volume Increase (x{vol_ratio:.1f})")
                elif vol_ratio < 0.5 and last['close'] > last['open']:
                    score_modifier -= 3.0
                    reasons.append("📉 Volumeless Rise (Fake?)")
            
            # D. MA Distance
            sma20 = last.get('SMA_20', np.mean(closes[-20:]) if len(closes) > 20 else closes[-1])
            if sma20 > 0:
                dist_sma = (last['close'] - sma20) / sma20 * 100
                safe_max = settings.SMART_ENTRY_MA_SAFE_MAX
                
                if dist_sma > safe_max:
                    score_modifier -= 5.0
                    reasons.append(f"⚠️ Too Far from MA20 ({dist_sma:.1f}%)")
                elif 0 < dist_sma < 2.0:
                    score_modifier += 2.5
                    reasons.append("✅ Close to MA20 Support")

            return score_modifier, reasons

        except Exception as e:
            logger.warning(f"Trend health check error: {e}")
            return 0.0, []

    def analyze_potential_entry(self, symbol: str, df: Any) -> Optional[Dict]:

        # --- 0. GENERAL MARKET PROTECTION (BTC FILTER) ---
        # Get BTC trend from market sentiment
        btc_trend = self.sentiment.btc_trend
        
        # If looking for Short trade and BTC is very strong, cancel analysis.
        if settings.SHORT_SELL and not getattr(settings, 'SKIP_BTC_TREND_FILTER', False):
            # If BTC is in Bullish mode, opening Short is suicide.
            if btc_trend in ['BULLISH', 'STRONG_BULLISH']:
                return None 
        # -----------------------------------------------

        # 1. Calculate Indicators
        df = AdvancedTechnicalAnalysis.calculate_enhanced_indicators(df)
        if 'RSI' not in df.columns: return None
        df = df.dropna()
        if df.empty or len(df) < 2: return None

        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        score_components = {}
        
        # ---------------------------------------------------------
        # 1. COMPONENT SCORES (Component Calculation)
        # ---------------------------------------------------------

        # --- 1. AUTOMATIC DIRECTION SELECTION (Auto-Pilot) ---
        if settings.FOLLOW_MARKET_REGIME:
            allowed_sides = self.sentiment.get_allowed_sides()
        else:
            allowed_sides = ['LONG', 'SHORT']
        
        # --- RSI ---
        rsi = last['RSI']
        # Long Logic
        if rsi > 70: score_components['rsi_long'] = -3.0
        elif rsi > 60: score_components['rsi_long'] = -2.0
        elif 25 < rsi < 35: score_components['rsi_long'] = 3.0
        elif 35 <= rsi < 45: score_components['rsi_long'] = 1.5
        elif 45 <= rsi <= 55: score_components['rsi_long'] = 0.0
        else: score_components['rsi_long'] = -1.0
        # Short Logic (Logic in spaghetti code)
        if rsi < 30: score_components['rsi_short'] = 3.0
        elif rsi < 40: score_components['rsi_short'] = 2.0
        elif 65 < rsi < 75: score_components['rsi_short'] = -3.0
        elif 55 <= rsi < 65: score_components['rsi_short'] = -1.5
        elif 45 <= rsi <= 55: score_components['rsi_short'] = 0.0
        else: score_components['rsi_short'] = 1.0

        # --- MACD ---
        macd_diff = last.get('MACD', 0) - last.get('MACD_Signal', 0)
        prev_macd_diff = prev.get('MACD', 0) - prev.get('MACD_Signal', 0)
        # Long Logic
        if macd_diff > 0 and prev_macd_diff <= 0: score_components['macd_long'] = 4.0
        elif macd_diff > 0: score_components['macd_long'] = 1.5
        elif macd_diff < 0 and prev_macd_diff >= 0: score_components['macd_long'] = -2.0
        else: score_components['macd_long'] = -0.5
        # Short Logic
        if macd_diff < 0 and prev_macd_diff >= 0: score_components['macd_short'] = -4.0
        elif macd_diff < 0: score_components['macd_short'] = -1.5
        elif macd_diff > 0 and prev_macd_diff <= 0: score_components['macd_short'] = 2.0
        else: score_components['macd_short'] = 0.5

        # --- Bollinger Bands ---
        bb_pos = 0.5
        if 'BB_Upper' in last and 'BB_Lower' in last:
             width = last['BB_Upper'] - last['BB_Lower']
             if width > 0:
                 bb_pos = (last['close'] - last['BB_Lower']) / width
        
        # Long
        if bb_pos < 0.15: score_components['bollinger_long'] = 3.0
        elif bb_pos < 0.3: score_components['bollinger_long'] = 1.5
        elif bb_pos > 0.85: score_components['bollinger_long'] = -2.0
        else: score_components['bollinger_long'] = 0.0
        # Short
        if bb_pos > 0.85: score_components['bollinger_short'] = -3.0
        elif bb_pos > 0.7: score_components['bollinger_short'] = -1.5
        elif bb_pos < 0.15: score_components['bollinger_short'] = 2.0
        else: score_components['bollinger_short'] = 0.0

        # --- Volume ---
        vol_ratio = last.get('Volume_Ratio', 1.0)
        vol_score = 3.0 if vol_ratio > 2.5 else (1.5 if vol_ratio > 1.5 else 0.0)
        score_components['volume_long'] = vol_score
        score_components['volume_short'] = -vol_score # High volume rise is bad for short

        # ---------------------------------------------------------
        # 2. RAW SCORE CALCULATION
        # ---------------------------------------------------------
        long_score = sum(v for k, v in score_components.items() if k.endswith('_long'))
        short_score = sum(v for k, v in score_components.items() if k.endswith('_short'))
        
        # Extra Short Calculations (Detailed negative points from Analysis.py)
        extra_technical_scores = AdvancedTechnicalAnalysis.calculate_short_score_detailed(df)
        raw_short_score_extra = extra_technical_scores['total']
        short_score -= raw_short_score_extra 

        # Sentiment Effect
        long_score += self.sentiment.sentiment_score
        short_score += self.sentiment.get_short_bias()

        # Whitelist
        whitelist_score = 0.0
        if symbol in settings.PREFERRED_SYMBOLS:
            whitelist_score = 1.0
            long_score += whitelist_score
            short_score -= whitelist_score

        # ---------------------------------------------------------
        # 3. TREND HEALTH CHECK (CRITICAL FIX LOCATION)
        # ---------------------------------------------------------
        health_score, health_reasons = self.calculate_trend_health(df)
        
        # A) Hard Filter (-100)
        if health_score <= -50.0:
            long_score = -999.0
            short_score = 999.0 # Giving high positive value to block Short
        else:
            # B) Soft Filter (Adding Points)
            long_score += health_score
            short_score += health_score 

        # ---------------------------------------------------------
        # 4. DETAILED SCORE MAP FOR DB (Mapping)
        # ---------------------------------------------------------
        db_scores = {
            'score_sentiment': self.sentiment.sentiment_score,
            'score_trend_health': health_score,
            'score_volume': 0.0, # Will be assigned below
            'score_patterns': 0.0, 'score_bonus': 0.0,
            'score_rsi': 0.0, 'score_macd': 0.0, 'score_bb': 0.0,
            'score_stoch': 0.0, 'score_williams': 0.0, 'score_adx': 0.0, 'score_ema': 0.0
        }

        # We will write scores of decided direction to DB
        # Note: Subtracting Analysis.py scores for Short (Net Effect)
        
        final_score = 0.0
        decision_type = None

        if 'SHORT' in allowed_sides and short_score <= settings.SHORT_SCORE_THRESHOLD:
            decision_type = 'SHORT'
            final_score = short_score
            
            # Volume for Short: Strategy Score - Analysis Score (MFI/Climax penalty)
            db_scores['score_volume'] = score_components['volume_short'] - extra_technical_scores.get('volume', 0)
            
            # FIX: Save sentiment score as Short Bias (e.g. -2.0 instead of 2.0)
            db_scores['score_sentiment'] = self.sentiment.get_short_bias()
            
            # For Short: Strategy Score - Analysis Score
            db_scores['score_rsi'] = score_components['rsi_short'] - extra_technical_scores.get('rsi', 0)
            db_scores['score_macd'] = score_components['macd_short'] - extra_technical_scores.get('macd', 0)
            db_scores['score_bb'] = score_components['bollinger_short'] - extra_technical_scores.get('bb', 0)
            # Other scores from Analysis.py (Reflecting as negative effect)
            db_scores['score_stoch'] = -extra_technical_scores.get('stoch', 0)
            db_scores['score_williams'] = -extra_technical_scores.get('williams', 0)
            db_scores['score_adx'] = -extra_technical_scores.get('adx', 0)
            db_scores['score_ema'] = -extra_technical_scores.get('ema', 0)
            
            # NEW: Patterns and Whitelist (Bonus)
            db_scores['score_patterns'] = -extra_technical_scores.get('patterns', 0)
            db_scores['score_bonus'] = -whitelist_score # Whitelist is penalty for Short (-1.0)
            
        elif 'LONG' in allowed_sides and long_score >= settings.MIN_SCORE_REQUIREMENT:
            decision_type = 'LONG'
            final_score = long_score
            db_scores['score_volume'] = score_components['volume_long']
            db_scores['score_rsi'] = score_components['rsi_long']
            db_scores['score_macd'] = score_components['macd_long']
            db_scores['score_bb'] = score_components['bollinger_long']
            db_scores['score_bonus'] = whitelist_score # Whitelist is a bonus for Long (+1.0)
            # Analysis.py scores not used for Long, remaining 0.

        db_scores['total'] = final_score

        # ---------------------------------------------------------
        # 5. RETURNING RESULT
        # ---------------------------------------------------------
        if decision_type:
            return {
                'symbol': symbol,
                'type': decision_type,
                'score': final_score,
                'current_price': last['close'],
                'volatility': last.get('NATR', 5.0),
                'detailed_scores': db_scores, # <-- NOW COMING FULL HERE
                'health_reasons': health_reasons,
                'short': (decision_type == 'SHORT')
            }

        return None

    def calculate_targets(self, analysis: Dict) -> Tuple[float, float]:
        """Dynamic TP and SL determination"""
        is_short = analysis.get('short', False)
        volatility = analysis.get('volatility', 5.0)
        
        # Volatility multiplier
        vol_mult = 1.0 + min(volatility / 25.0, 0.4)
        
        # Leverage Based Maximum Safe Stop (Liquidation Protection)
        # E.g. 10x leverage means 10% liquidation. We accept 80% of it (i.e. 8%) as limit.
        max_safe_sl = -(80.0 / settings.LEVERAGE)
        
        if is_short:
            base_profit = 4.0
            base_stop = -8.0
            
            # The lower the short score (negative), the safer it is
            short_score = abs(analysis.get('score', 0))
            score_mult = 1.0 + (short_score / 15.0)
            
            # Sentiment Effect
            sentiment_score = self.sentiment.get_short_bias()
            if sentiment_score < -5: sentiment_mult = 1.3
            elif sentiment_score < -3: sentiment_mult = 1.15
            else: sentiment_mult = 1.0
            
            tp = base_profit * vol_mult * score_mult * sentiment_mult
            sl = base_stop * max(vol_mult, 1.0) # FIX: Stop should widen as volatility increases
            sl = max(sl, max_safe_sl) # Liquidation protection
            # Limits
            tp = min(max(tp, 3.0), 12.0)
            sl = max(min(sl, -4.0), -15.0)
            
        else: # LONG
            base_profit = 5.0
            base_stop = -6.0
            score = analysis.get('score', 0)
            score_mult = 1.0 + ((score - settings.MIN_SCORE_REQUIREMENT) / 40.0)
            
            tp = base_profit * vol_mult * score_mult
            sl = base_stop * max(vol_mult, 1.0) # FIX: Stop should widen as volatility increases
            sl = max(sl, max_safe_sl) # Liquidation protection
            
            tp = min(max(tp, 3.0), 15.0)
            sl = max(min(sl, -3.0), -12.0)

        # Risk/Reward Check
        risk_reward = abs(tp / sl)
        min_rr = 1.3 if is_short else 1.5
        if risk_reward < min_rr:
            tp = abs(sl) * min_rr

        return round(tp, 2), round(sl, 2)

    def check_exit_conditions(self, position: Dict, current_price: float, df: pd.DataFrame) -> Tuple[bool, str]:
        """Decision to close position or not"""
        
        buy_price = position['buy_price']
        pos_type = position.get('position_type', 'LONG')
        buy_time = position['buy_time']
        leverage = position.get('leverage', 1)
        
        # Calculate PnL
        if pos_type == 'SHORT':
            profit_pct = ((buy_price - current_price) / buy_price) * 100
        else:
            profit_pct = ((current_price - buy_price) / buy_price) * 100
            
        roi = profit_pct * leverage
        hold_time_hours = (datetime.now() - buy_time).total_seconds() / 3600

        # --- NEW: Trailing Stop Check (Priority Check) ---
        trailing_stop_price = position.get('trailing_stop_price')
        if trailing_stop_price:
            if pos_type == 'LONG' and current_price <= trailing_stop_price:
                return True, f"📉 Trailing Stop Triggered (Price: {current_price:.4f})"
            elif pos_type == 'SHORT' and current_price >= trailing_stop_price:
                return True, f"📈 Trailing Stop Triggered (Price: {current_price:.4f})"
        # --------------------------------------------------------

        # 1. Fixed Targets (TP / SL)
        if profit_pct >= position['profit_target']:
            return True, f"🎯 Take Profit ({profit_pct:.2f}%)"
        if profit_pct <= position['stop_loss']:
            return True, f"🛑 Stop Loss ({profit_pct:.2f}%)"
        
        # 2. Emergency (Hard Stop)
        if profit_pct <= -settings.MAX_ALLOWED_LOSS_ON_EXIT:
             return True, f"⚠️ Emergency Exit Limit ({profit_pct:.2f}%)"

        # 3. Profit Retention - NEW
        # If we saw more than 20% profit before and gave back 40% of it, exit.
        # E.g. Saw Max 22% -> Sell if drops to 13.2%.
        max_roi = position.get('max_profit_pct', 0.0)
        if max_roi > 20.0 and roi < (max_roi * 0.6):
            return True, f"📉 Profit Erosion Protection (Max: %{max_roi:.1f} -> Active: %{roi:.1f})"

        # 4. Time Based Exits
        if hold_time_hours > settings.MAX_HOLD_TIME_DAYS * 24:
            return True, "⏰ Maximum holding time expired"
        if hold_time_hours > 6 and profit_pct > 10.0:
            return True, "⏰ Timeout + Profit"

        # 5. Indicator Based Dynamic Exits
        if not df.empty:
            df = AdvancedTechnicalAnalysis.calculate_enhanced_indicators(df)
            if 'RSI' not in df.columns: return False, ""

            last = df.iloc[-1]
            
            # Getting current settings from settings.py
            min_profit = settings.MIN_PROFIT_FOR_INDICATOR_EXIT  # E.g.: 3.5
            rsi_long_limit = settings.RSI_EXIT_THRESHOLD_LONG    # E.g.: 82
            rsi_short_limit = settings.RSI_EXIT_THRESHOLD_SHORT  # E.g.: 18

            if pos_type == 'LONG':
                # Now expecting RSI 82 instead of 78 AND profit must be at least 3.5%
                if last['RSI'] > rsi_long_limit and profit_pct > min_profit:
                    return True, f"📊 RSI Overbought ({last['RSI']:.1f})"
                
                # Even if MACD crosses, won't sell if profit under 3.5%, rides trend
                if last.get('MACD', 0) < last.get('MACD_Signal', 0) and profit_pct > min_profit:
                    return True, "📉 MACD Bearish Cross"
                
                # Volume drop is serious signal, profit threshold higher (8%) can be maintained here
                if last.get('Volume_Ratio', 1) < 0.5 and profit_pct > min_profit:
                    return True, "📉 Volume Drop"
                    
            elif pos_type == 'SHORT':
                # Waits until RSI drops to 18 instead of 25 in Short trade
                if last['RSI'] < rsi_short_limit and profit_pct > min_profit:
                    return True, f"📊 RSI Oversold ({last['RSI']:.1f})"
                
                if last.get('MACD', 0) > last.get('MACD_Signal', 0) and profit_pct > min_profit:
                    return True, "📈 MACD Bullish Cross (Close Short)"
                
                if last.get('Volume_Ratio', 1) > 2.5 and last['close'] > last['open'] and profit_pct > min_profit:
                    return True, "📈 Strong Buy Volume (Close Short)"
                
                # Liquidity Risk Check
                liq_price = position.get('liquidation_price', 0)
                if liq_price > 0:
                    dist = ((liq_price - current_price) / current_price) * 100
                    if dist > 15: # If approaches more than 15%
                        return True, f"🚨 Liquidity Risk ({dist:.1f}%)"

        return False, ""

    def update_trailing_stop(self, position: Dict, current_price: float, df: pd.DataFrame) -> Dict:
        """Updates trailing stop level"""
        if df.empty: return position
        
        try:
            last_atr = df.iloc[-1]['ATR']
            natr = df.iloc[-1].get('NATR', 2.0) # Volatility data
        except:
            last_atr = current_price * 0.02
            natr = 2.0

        pos_type = position.get('position_type', 'LONG')
        leverage = position.get('leverage', 1)
        buy_price = position['buy_price']
        
        # Calculate PnL (Unleveraged)
        if pos_type == 'SHORT':
            profit_pct = ((buy_price - current_price) / buy_price) * 100
        else:
            profit_pct = ((current_price - buy_price) / buy_price) * 100
            
        roi = profit_pct * leverage

        # --- DYNAMIC ATR MULTIPLIER (Based on Volatility and Profit) ---
        atr_multiplier = 1.5
        
        # 1. Narrowing by volatility (High volatility -> Tight tracking)
        if natr > 4.0: atr_multiplier = 0.8
        elif natr > 2.5: atr_multiplier = 1.2
        
        # 2. Narrowing by profit (Profit Locking)
        if roi > 50: atr_multiplier = min(atr_multiplier, 0.5) # Very tight tracking over 50% profit
        elif roi > 30: atr_multiplier = min(atr_multiplier, 0.8) # Tightened (1.0 -> 0.8)
        elif roi > 15: atr_multiplier = min(atr_multiplier, 1.0) # NEW: Start tracking at 15% profit
        # -----------------------------------------------------

        if pos_type == 'LONG':
            # Activation: 1.0% raw price movement
            if profit_pct > 1.0: 
                new_stop = current_price - (last_atr * atr_multiplier)
                
                # Assign if no initial trailing stop
                if 'trailing_stop_price' not in position:
                     position['trailing_stop_price'] = new_stop
                     position['highest_price'] = current_price
                else:
                    # Pull stop up as price rises
                    if current_price > position.get('highest_price', 0):
                        position['highest_price'] = current_price
                        if new_stop > position['trailing_stop_price']:
                            position['trailing_stop_price'] = new_stop
            
            # --- BREAKEVEN MECHANISM ---
            # If ROI exceeds 12% (Old 20%), pull Stop Loss to at least entry level (+commission)
            if roi > 12.0:
                breakeven_price = buy_price * 1.002 
                if position.get('trailing_stop_price', 0) < breakeven_price:
                    position['trailing_stop_price'] = breakeven_price
                    
        elif pos_type == 'SHORT':
            if profit_pct > 1.5:
                new_stop = current_price + (last_atr * atr_multiplier)
                
                if 'trailing_stop_price' not in position:
                    position['trailing_stop_price'] = new_stop
                    position['lowest_price'] = current_price
                else:
                    # Pull stop down as price falls
                    if current_price < position.get('lowest_price', float('inf')):
                        position['lowest_price'] = current_price
                        if new_stop < position['trailing_stop_price']:
                            position['trailing_stop_price'] = new_stop

            # --- BREAKEVEN MECHANISM ---
            if roi > 12.0:
                breakeven_price = buy_price * 0.998
                if position.get('trailing_stop_price', float('inf')) > breakeven_price:
                    position['trailing_stop_price'] = breakeven_price
                    
        return position