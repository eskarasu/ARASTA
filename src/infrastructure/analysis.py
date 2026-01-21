import pandas as pd
import numpy as np
import talib
from typing import Dict, Optional
from src.infrastructure.logger import logger

class AdvancedTechnicalAnalysis:
    """
    Advanced Technical Analysis Engine.
    Implements Aggressive Momentum Strategy with Clean Code.
    Includes Late Entry and Momentum Recovery protections.
    """
    
    @staticmethod
    def calculate_enhanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates advanced indicators (TA-Lib Supported)"""
        if df.empty or len(df) < 50:
            return df
        
        try:
            # Ensure Data Types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # --- 1. Basic Indicators (RSI) ---
            try:
                df['RSI'] = talib.RSI(close, timeperiod=14)
                df['RSI_Fast'] = talib.RSI(close, timeperiod=7)
                df['RSI_Slow'] = talib.RSI(close, timeperiod=21)
            except Exception:
                df['RSI'] = 50.0
                df['RSI_Fast'] = df['RSI']
                df['RSI_Slow'] = df['RSI']

            # --- 2. MACD ---
            try:
                df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(
                    close, fastperiod=12, slowperiod=26, signalperiod=9
                )
                # Fast MACD for faster reaction
                df['MACD_Fast'], _, _ = talib.MACD(
                    close, fastperiod=6, slowperiod=13, signalperiod=5
                )
            except Exception:
                df['MACD'] = 0.0; df['MACD_Signal'] = 0.0

            # --- 3. Bollinger Bands ---
            try:
                df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(
                    close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
                )
                # BB Width (Band Width)
                with np.errstate(divide='ignore', invalid='ignore'):
                    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] * 100
                df['BB_Width'] = df['BB_Width'].replace([np.inf, -np.inf], 0.0).fillna(0.0)
            except Exception: pass

            # --- 4. Oscillators ---
            try:
                df['Williams_R'] = talib.WILLR(high, low, close, timeperiod=14)
                df['Stoch_K'], df['Stoch_D'] = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
                
                # NEW: MFI (Money Flow Index) - Volume weighted RSI
                df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
                
            except Exception: pass

            # --- 5. ADX (Trend Strength) ---
            try:
                df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
                df['DI_Plus'] = talib.PLUS_DI(high, low, close, timeperiod=14)
                df['DI_Minus'] = talib.MINUS_DI(high, low, close, timeperiod=14)
            except Exception: df['ADX'] = 0.0

            # --- 6. Momentum ---
            try:
                df['MOM'] = talib.MOM(close, timeperiod=10)
                df['ROC'] = talib.ROC(close, timeperiod=10)
            except Exception: pass

            # --- 7. Volume Indicators ---
            try:
                df['OBV'] = talib.OBV(close, volume)
                df['AD'] = talib.AD(high, low, close, volume)
            except Exception: pass

            # --- 8. Volatility ---
            try:
                df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
                df['NATR'] = talib.NATR(high, low, close, timeperiod=14)
            except Exception: pass

            # --- 9. Moving Averages ---
            try:
                df['EMA_9'] = talib.EMA(close, timeperiod=9)
                df['EMA_21'] = talib.EMA(close, timeperiod=21)
                df['EMA_50'] = talib.EMA(close, timeperiod=50)
                df['SMA_20'] = talib.SMA(close, timeperiod=20)
            except Exception: pass

            # --- 10. Custom Derived Indicators ---
            
            # Price Position (Price position relative to SMA20 / in ATR)
            if 'SMA_20' in df.columns and 'ATR' in df.columns:
                safe_atr = df['ATR'].replace(0, np.nan)
                with np.errstate(divide='ignore', invalid='ignore'):
                    df['Price_Position'] = (df['close'] - df['SMA_20']) / safe_atr
                df['Price_Position'] = df['Price_Position'].fillna(0.0)
            
            # Volume Ratio (Instant Volume / Avg. Volume)
            try:
                sma_vol = talib.SMA(volume, timeperiod=20)
                with np.errstate(divide='ignore', invalid='ignore'):
                    vol_ratio = volume / sma_vol
                df['Volume_Ratio'] = vol_ratio
                df['Volume_Ratio'] = df['Volume_Ratio'].replace([np.inf, -np.inf], 0.0).fillna(0.0)
            except Exception: df['Volume_Ratio'] = 0.0

            # Trend Score (Simple direction determination)
            df['Trend_Score'] = 0
            if 'EMA_21' in df.columns: df.loc[df['close'] > df['EMA_21'], 'Trend_Score'] += 1
            if 'MACD' in df.columns: df.loc[df['MACD'] > df['MACD_Signal'], 'Trend_Score'] += 1
            if 'ADX' in df.columns: df.loc[df['ADX'] > 25, 'Trend_Score'] += 1
            if 'DI_Plus' in df.columns: df.loc[df['DI_Plus'] > df['DI_Minus'], 'Trend_Score'] += 1
            
        except Exception as e:
            logger.warning(f"Error in technical indicator calculation: {e}")
        
        return df

    @staticmethod
    def calculate_momentum_score(df: pd.DataFrame) -> float:
        """Momentum score calculation (Used for Long trades)"""
        if df.empty or len(df) < 10: return 0.0
        try:
            last = df.iloc[-1]; prev = df.iloc[-2]
            score = 0.0

            def safe_get(row, key, default=0.0):
                try: return float(row.get(key, default))
                except: return float(default)

            if safe_get(last, 'RSI') > safe_get(prev, 'RSI'): score += 1.0
            if safe_get(last, 'RSI_Fast') > 50: score += 0.5
            if safe_get(last, 'MACD') > safe_get(last, 'MACD_Signal'):
                score += 1.0
                if safe_get(prev, 'MACD') <= safe_get(prev, 'MACD_Signal'): score += 2.0
            if safe_get(last, 'Stoch_K') > safe_get(last, 'Stoch_D'): score += 0.5
            if safe_get(last, 'ADX') > 25 and safe_get(last, 'DI_Plus') > safe_get(last, 'DI_Minus'): score += 1.5
            
            ema9 = last.get('EMA_9'); ema21 = last.get('EMA_21')
            if ema9 and ema21 and float(last['close']) > float(ema9) > float(ema21):
                score += 1.0

            if safe_get(last, 'Williams_R', -100) > -80: score += 0.5
            
            return min(score, 8.0)
        except Exception as e:
            logger.warning(f"Momentum score error: {e}")
            return 0.0

    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> Dict[str, bool]:
        """BULLISH Patterns (Long Signals)"""
        patterns = {'bullish_engulfing': False, 'hammer': False, 'doji': False, 'breakout': False, 'support_bounce': False, 'volume_spike': False}
        if df.empty or len(df) < 20: return patterns
        try:
            last = df.iloc[-1]; prev = df.iloc[-2]
            body = abs(last['close'] - last['open'])
            
            # Engulfing
            if (prev['close'] < prev['open'] and last['close'] > last['open'] and last['close'] > prev['open']):
                patterns['bullish_engulfing'] = True
            
            # Hammer
            lower = last['open'] - last['low'] if last['close'] > last['open'] else last['close'] - last['low']
            upper = last['high'] - max(last['open'], last['close'])
            if lower > body * 2 and upper < body: patterns['hammer'] = True
            
            # Breakout
            if last.get('BB_Upper') and last['close'] > last['BB_Upper']: patterns['breakout'] = True
            
            # Volume Spike
            try:
                avg_vol = df['volume'].iloc[-10:-1].mean()
                if last['volume'] > avg_vol * 2: patterns['volume_spike'] = True
            except: pass
        except: pass
        return patterns

    @staticmethod
    def detect_bearish_patterns(df: pd.DataFrame) -> Dict[str, bool]:
        """BEARISH Patterns (Short Signals)"""
        patterns = {'bearish_engulfing': False, 'shooting_star': False, 'resistance_rejection': False, 'volume_climax': False, 'macd_death_cross': False, 'rsi_divergence': False}
        if df.empty or len(df) < 20: return patterns
        try:
            last = df.iloc[-1]; prev = df.iloc[-2]
            body = abs(last['close'] - last['open'])
            
            # FIX: Trend check for reversal patterns
            # Reversal searched if price is above EMA 50 (Uptrend).
            is_uptrend = last['close'] > last.get('EMA_50', 0)

            # Engulfing (Stricter rules)
            # 1. Trend must be up (Reversal from top)
            # 2. Previous candle green, current red
            # 3. Current body must engulf previous body
            if (is_uptrend and 
                prev['close'] > prev['open'] and 
                last['close'] < last['open'] and 
                last['open'] >= prev['close'] and 
                last['close'] <= prev['open']):
                patterns['bearish_engulfing'] = True

            # Shooting Star
            upper = last['high'] - max(last['open'], last['close'])
            lower = min(last['open'], last['close']) - last['low']
            # Trend up + Long upper wick + Short lower wick
            if is_uptrend and upper > body * 2 and lower < body: 
                patterns['shooting_star'] = True

            # Resistance Rejection
            if last.get('BB_Upper') and last['close'] < last['BB_Upper'] and prev['close'] >= prev.get('BB_Upper', float('inf')):
                patterns['resistance_rejection'] = True

            # Volume Climax
            try:
                avg_vol = df['volume'].iloc[-10:-1].mean()
                if last['volume'] > avg_vol * 2.5 and last['close'] < last['open']:
                    patterns['volume_climax'] = True
            except: pass

            # MACD Death Cross
            if (last.get('MACD', 0) < last.get('MACD_Signal', 0) and prev.get('MACD', 0) >= prev.get('MACD_Signal', 0)):
                patterns['macd_death_cross'] = True

            # RSI Divergence
            if len(df) >= 10:
                prev_p = df.iloc[-10]['close']; prev_r = df.iloc[-10]['RSI']
                if prev_p != 0 and prev_r != 0 and not np.isnan(prev_r):
                    p_trend = (last['close'] - prev_p) / prev_p
                    r_trend = (last['RSI'] - prev_r) / prev_r
                    if p_trend > 0.02 and r_trend < -0.05: patterns['rsi_divergence'] = True

        except: pass
        return patterns

    @staticmethod
    def calculate_short_score_detailed(df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates SHORT score and returns details.
        NEW ADDED: Late Entry and Recovery Protections.
        """
        result = {'rsi': 0.0, 'macd': 0.0, 'stoch': 0.0, 'williams': 0.0, 'adx': 0.0, 'ema': 0.0, 'bb': 0.0, 'patterns': 0.0, 'volume': 0.0, 'total': 0.0}
        if df.empty or len(df) < 10: return result
        
        try:
            last = df.iloc[-1]; prev = df.iloc[-2]
            
            # Pre-calculations
            adx = last.get('ADX', 0)
            di_plus = last.get('DI_Plus', 0)
            di_minus = last.get('DI_Minus', 0)
            
            # --- 1. RSI (Late Entry Protected) ---
            rsi = last.get('RSI', 50)
            if rsi > 85: 
                result['rsi'] = 2.5      # Extreme overbought
            elif rsi > 70: 
                # Don't trust RSI if trend is too strong (Parabolic run)
                if adx > 35 and di_plus > di_minus:
                    result['rsi'] = 0.0
                else:
                    result['rsi'] = 2.0
            elif rsi < 35: 
                result['rsi'] = -3.0   # Oversold short penalty
            
            # NEW: MFI Check (Volume Supported Bloat)
            mfi = last.get('MFI', 50)
            if mfi > 80: result['volume'] += 1.5
            elif mfi > 70: result['volume'] += 0.5
            
            # PROTECTION 1: RSI Recovery Detector
            if len(df) >= 5:
                rsi_5_ago = df.iloc[-5]['RSI']
                rsi_trend = rsi - rsi_5_ago
                if rsi < 30 and rsi_trend > 5:
                    result['rsi'] = -5.0 

            # --- 2. MACD (Momentum Protected) ---
            macd_diff = last.get('MACD', 0) - last.get('MACD_Signal', 0)
            prev_macd_diff = prev.get('MACD', 0) - prev.get('MACD_Signal', 0)
            
            if macd_diff < 0 and macd_diff < prev_macd_diff: result['macd'] = 2.0 # Decline strengthening
            elif macd_diff < -0.5: result['macd'] = 1.0
            
            # PROTECTION 2: MACD Momentum Detector
            if len(df) >= 5:
                macd_5_ago_diff = df.iloc[-5]['MACD'] - df.iloc[-5]['MACD_Signal']
                macd_momentum = macd_diff - macd_5_ago_diff
                if macd_diff < 0 and macd_momentum > 0:
                    result['macd'] = -2.0 # Penalty

            # --- 3. Stochastic ---
            if last.get('Stoch_K', 50) > 85: result['stoch'] = 1.5

            # --- 4. Williams %R ---
            if last.get('Williams_R', -50) > -10: result['williams'] = 1.5 # Extreme overbought
            elif last.get('Williams_R', -50) > -20: result['williams'] = 1.0
            
            # --- 5. ADX (Trend Strength - Parabolic Protection) ---
            # FIX: Prevent opening Short in Strong Bull Trend (Parabolic Protection)
            if adx > 30 and di_plus > di_minus:
                result['adx'] = -10.0 # Penalty (Harsher penalty: -5 -> -10)
            elif adx > 25 and di_minus > di_plus: 
                result['adx'] = 2.0
            
            # --- 6. EMA Ranking ---
            if (last.get('EMA_9', 0) < last.get('EMA_21', 0) and last.get('EMA_21', 0) < last.get('EMA_50', 0)): 
                result['ema'] = 1.5

            # --- 7. Bollinger Bands (Recovery Protected) ---
            bb_position = 0.0
            try:
                denom = last['BB_Upper'] - last['BB_Lower']
                if denom > 0: bb_position = (last['close'] - last['BB_Lower']) / denom
            except: pass
            
            if bb_position > 0.9: result['bb'] = 2.0
            
            # PROTECTION 3: BB Lower Band Protection (Shorting in the hole)
            # Do not open short if price is below or very close to BB Lower band
            if bb_position < 0.05:
                result['bb'] = -4.0 # Penalty
            elif bb_position < 0.2:
                # Is there a reversal from bottom?
                if len(df) >= 3:
                    row_3 = df.iloc[-3]
                    denom_3 = row_3['BB_Upper'] - row_3['BB_Lower']
                    if denom_3 > 0:
                        prev_bb_pos = (row_3['close'] - row_3['BB_Lower']) / denom_3
                        if prev_bb_pos < bb_position:
                            result['bb'] = -3.0 # Penalty

            # --- 8. Patterns (NEW) ---
            patterns = AdvancedTechnicalAnalysis.detect_bearish_patterns(df)
            if patterns.get('bearish_engulfing'): result['patterns'] += 2.5
            if patterns.get('shooting_star'): result['patterns'] += 2.0
            if patterns.get('resistance_rejection'): result['patterns'] += 1.5
            if patterns.get('volume_climax'): result['volume'] += 1.5
            
            # Add pattern score to total (but also keep as separate key)
            # patterns will be included when calculating result['total'].
            # FIX: Do not include 'total' key in summation (Prevent double counting)
            raw_total = sum(v for k, v in result.items() if k != 'total')
            result['total'] = min(raw_total, 10.0)
            
            return result
            
        except Exception as e:
            logger.warning(f"Detailed score calculation error: {e}")
            return result

    @staticmethod
    def calculate_short_score(df: pd.DataFrame) -> float:
        """Wrapper for legacy code compatibility"""
        res = AdvancedTechnicalAnalysis.calculate_short_score_detailed(df)
        return res['total']

    @staticmethod
    def calculate_long_score_detailed(df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates LONG score and returns details.
        Includes 'Falling Knife' and 'Fakeout' protections for more profitable results.
        """
        result = {'rsi': 0.0, 'macd': 0.0, 'stoch': 0.0, 'williams': 0.0, 'adx': 0.0, 'ema': 0.0, 'bb': 0.0, 'patterns': 0.0, 'volume': 0.0, 'total': 0.0}
        if df.empty or len(df) < 20: return result
        
        try:
            last = df.iloc[-1]; prev = df.iloc[-2]
            
            # --- 1. EMA Trend (GOLDEN RULE) ---
            # Opening Long is very risky if price is below EMA 200.
            ema50 = last.get('EMA_50', 0)
            ema200 = last.get('EMA_200', 0)
            close = last['close']
            
            if close > ema50 and ema50 > ema200: result['ema'] = 3.0  # Strong Uptrend
            elif close > ema200: result['ema'] = 1.0                  # Uptrend Zone
            elif close < ema50 and close < ema200: result['ema'] = -8.0 # Downtrend (PROTECTION) -5 -> -8

            # --- 2. ADX (Trend Strength and Direction) ---
            adx = last.get('ADX', 0)
            di_plus = last.get('DI_Plus', 0)
            di_minus = last.get('DI_Minus', 0)
            
            if adx > 25:
                if di_plus > di_minus: result['adx'] = 2.0  # Strong Buy Trend
                elif di_minus > di_plus: result['adx'] = -4.0 # Strong Sell Trend (Long Forbidden)

            # --- 3. RSI (Momentum and Bottom) ---
            rsi = last.get('RSI', 50)
            if rsi > 70: result['rsi'] = -2.0       # Overbought (Risk of buying at top)
            elif 50 <= rsi <= 70: result['rsi'] = 2.0 # Momentum Behind You
            elif 40 <= rsi < 50: result['rsi'] = 1.0  # Recovery Zone
            elif rsi < 30: result['rsi'] = 1.5        # Oversold Reaction (Risky but can be profitable)

            # --- 4. MACD (Crossover) ---
            macd = last.get('MACD', 0)
            signal = last.get('MACD_Signal', 0)
            if macd > signal:
                result['macd'] = 2.0
                if last.get('MACD_Hist', 0) > prev.get('MACD_Hist', 0): result['macd'] += 1.0 # Momentum increasing
            else:
                result['macd'] = -1.5 # Selling pressure

            # --- 5. Bollinger Bands (Position) ---
            bb_pos = 0.5
            if 'BB_Upper' in last and 'BB_Lower' in last:
                width = last['BB_Upper'] - last['BB_Lower']
                if width > 0: bb_pos = (close - last['BB_Lower']) / width
            
            if bb_pos > 0.95: result['bb'] = -3.0 # Stuck to band top (Correction risk)
            elif bb_pos < 0.2: result['bb'] = 2.0 # At band bottom (Cheap)
            
            # --- 6. Volume and MFI ---
            mfi = last.get('MFI', 50)
            if mfi < 20: result['volume'] += 2.0 # Money outflow stopped, buying opportunity
            
            # --- TOTAL SCORE ---
            # Do not include 'total' key in summation
            raw_total = sum(v for k, v in result.items() if k != 'total')
            result['total'] = min(max(raw_total, -10.0), 15.0)
            
            return result
        except Exception as e:
            logger.warning(f"Long score calculation error: {e}")
            return result
