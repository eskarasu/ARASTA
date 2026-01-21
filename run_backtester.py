import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.application.backtester import BacktestEngine
from src.infrastructure.binance_api import BinanceAdapter

def find_similar_historical_period(days=14, lookback_days=1000):
    """
    Finds the historical period most similar to current market conditions (last N days).
    Similarity Criteria: Shape of normalized price movement (Pattern Matching).
    """
    print(f"🔍 Market Regime Analysis: Searching for period most similar to last {days} days...")
    
    adapter = BinanceAdapter()
    # Fetch data (Daily candles sufficient)
    df = adapter.get_klines("BTCUSDT", "1d", limit=lookback_days)
    
    if df.empty or len(df) < days * 2:
        print("⚠️ Insufficient data, default dates will be used.")
        return None, None

    # Get close prices
    closes = df['close'].values
    dates = df['timestamp'].dt.strftime('%Y-%m-%d').values

    # Current period (Target)
    current_window = closes[-days:]
    
    # Normalize (Compress between 0 and 1 so price difference doesn't affect, only shape matters)
    def normalize(arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-9)

    target_norm = normalize(current_window)
    
    best_score = float('inf')
    best_start_idx = 0
    
    # Scan History (Excluding last 14 days, as it matches itself)
    # Go back at least 30 days so it doesn't find 'today'.
    search_range = len(closes) - days - 30
    
    for i in range(search_range):
        window = closes[i : i + days]
        window_norm = normalize(window)
        
        # Euclidean Distance - The smaller, the more similar
        score = np.linalg.norm(target_norm - window_norm)
        
        if score < best_score:
            best_score = score
            best_start_idx = i

    # Find best matching dates
    match_start_date = dates[best_start_idx]
    match_end_date = dates[best_start_idx + days]
    
    # Similarity Ratio (Invert for visualization)
    similarity_pct = max(0, 100 - (best_score * 10)) # Rough scoring
    
    print(f"✅ MATCH FOUND!")
    print(f"   Current Period : {dates[-days]} -> {dates[-1]}")
    print(f"   Similar Period : {match_start_date} -> {match_end_date}")
    print(f"   Similarity     : %{similarity_pct:.2f} (Score: {best_score:.4f})")
    print("-" * 50)
    
    return match_start_date, match_end_date

def main():
    # --- SETTINGS ---
    
    # 1. Automatic Date Finder
    sim_start, sim_end = find_similar_historical_period(days=14)
    
    if sim_start and sim_end:
        START_DATE = sim_start
        END_DATE = sim_end
    else:
        # Fallback (Manual date if not found)
        START_DATE = "2023-10-15"
        END_DATE = "2023-10-29"
        
    INTERVAL = "1h"
    
    # Symbols to test
    symbols = [
        # --- 1. MAJORS ---
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'LINKUSDT', 'ALGOUSDT',
        
        # --- 2. MEME COINS ---
        'DOGEUSDT', '1000PEPEUSDT', '1000SHIBUSDT', 'WIFUSDT', 
        '1000BONKUSDT', '1000FLOKIUSDT', 'BOMEUSDT', 'MEMEUSDT', 'PEOPLEUSDT',
        
        # --- 3. TRENDING TECH ---
        'FETUSDT', 'RENDERUSDT', 'NEARUSDT', 'TIAUSDT', 'INJUSDT',
        'GRTUSDT', 'WLDUSDT', 'ARKMUSDT', 'RUNEUSDT',
        
        # --- 4. NEW & HOT ---
        'SUIUSDT', 'SEIUSDT', 'APTUSDT', 'JUPUSDT', 'PYTHUSDT',
        'ENAUSDT', 'ETHFIUSDT', 'OMNIUSDT',
        
        # --- 5. PUMP CANDIDATES ---
        'TRBUSDT', 'GASUSDT', 'BLZUSDT', 'BIGTIMEUSDT', 'ORDIUSDT', 
        'LOOMUSDT', 'STORJUSDT',
        
        # --- 6. DEFI & LEGACY ---
        'AVAXUSDT', 'FTMUSDT', 'LTCUSDT',
        
        # --- 7. SPECIAL ---
        'RIVERUSDT', 'ZKPUSDT'
    ]
    
    # --- STATISTICS COLLECTORS ---
    grand_total_pnl = 0.0
    initial_total_capital = 0.0
    final_total_capital = 0.0
    
    # Detailed Statistics (For Long/Short Separation)
    stats = {
        'total': {'count': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0},
        'long':  {'count': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0},
        'short': {'count': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0}
    }
    
    print(f"🧪 Backtest Starting... ({START_DATE} - {END_DATE})\n")
    print(f"🌊 Market Regime Simulation: ACTIVE")
    print("-" * 50)
    
    for symbol in symbols:
        try:
            # Start Engine (BTC Trend Data downloaded here)
            engine = BacktestEngine(
                symbol=symbol, 
                interval=INTERVAL, 
                start_date=START_DATE, 
                end_date=END_DATE
            )
            engine.run()
            
            # General Capital Tracking
            initial_total_capital += engine.initial_balance
            final_total_capital += engine.balance
            grand_total_pnl += (engine.balance - engine.initial_balance)
            
            # Analyze Trades
            for trade in engine.trade_history:
                pnl = trade['pnl']
                is_win = pnl > 0
                t_type = trade['type'] # 'LONG' or 'SHORT'
                
                # Grand Total
                stats['total']['count'] += 1
                stats['total']['pnl'] += pnl
                if is_win: stats['total']['wins'] += 1
                else: stats['total']['losses'] += 1
                
                # Type Based Separation
                if t_type in ['LONG', 'SHORT']:
                    key = t_type.lower() # 'long' or 'short'
                    stats[key]['count'] += 1
                    stats[key]['pnl'] += pnl
                    if is_win: stats[key]['wins'] += 1
                    else: stats[key]['losses'] += 1
                    
        except Exception as e:
            print(f"Error ({symbol}): {e}")

    # --- HELPER FUNCTION: RATE CALCULATION ---
    def calc_wr(wins, total):
        return (wins / total * 100) if total > 0 else 0.0

    # --- PRINTING GENERAL REPORT ---
    print("\n" + "="*75)
    print(f"{'🌍 GENERAL PORTFOLIO REPORT (GRAND TOTAL)':^75}")
    print("="*75)
    print(f"Date Range            : {START_DATE} -> {END_DATE}")
    print(f"Tested Symbols        : {len(symbols)}")
    print(f"Total Initial Capital : {initial_total_capital:.2f} USDT")
    print(f"Total Final Capital   : {final_total_capital:.2f} USDT")
    
    # General ROI Calculation
    total_roi = ((final_total_capital - initial_total_capital) / initial_total_capital * 100) if initial_total_capital > 0 else 0
    icon = "✅" if grand_total_pnl > 0 else "❌"
    
    print("-" * 75)
    print(f"{icon} TOTAL NET PnL       : {grand_total_pnl:+.2f} USDT")
    print(f"📈 TOTAL GROWTH (ROI)  : %{total_roi:.2f}")
    print("="*75)
    
    print(f"\n📊 DETAILED PERFORMANCE ANALYSIS:")
    print(f"{'TYPE':<10} | {'TRADE':<8} | {'WIN RATE':<15} | {'WIN/LOSS':<15} | {'NET PnL ($)':<12}")
    print("-" * 75)
    
    # 1. LONG STATISTICS
    long_wr = calc_wr(stats['long']['wins'], stats['long']['count'])
    print(f"{'LONG':<10} | {stats['long']['count']:<8} | %{long_wr:<14.2f} | {stats['long']['wins']}/{stats['long']['losses']:<12} | {stats['long']['pnl']:+.2f}")
    
    # 2. SHORT STATISTICS
    short_wr = calc_wr(stats['short']['wins'], stats['short']['count'])
    print(f"{'SHORT':<10} | {stats['short']['count']:<8} | %{short_wr:<14.2f} | {stats['short']['wins']}/{stats['short']['losses']:<12} | {stats['short']['pnl']:+.2f}")
    
    print("-" * 75)
    
    # 3. TOTAL
    total_wr = calc_wr(stats['total']['wins'], stats['total']['count'])
    print(f"{'TOTAL':<10} | {stats['total']['count']:<8} | %{total_wr:<14.2f} | {stats['total']['wins']}/{stats['total']['losses']:<12} | {stats['total']['pnl']:+.2f}")
    print("="*75 + "\n")

if __name__ == "__main__":
    main()