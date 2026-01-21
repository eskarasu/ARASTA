import sys
import os

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.application.backtester import BacktestEngine

def main():
    # --- SETTINGS ---
    START_DATE = "2024-03-12"
    END_DATE = "2024-03-19"
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
    
    # Detailed Stats (Long/Short Separation)
    stats = {
        'total': {'count': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0},
        'long':  {'count': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0},
        'short': {'count': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0}
    }
    
    print(f"🧪 Starting Backtest... ({START_DATE} - {END_DATE})\n")
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

    # --- HELPER: RATIO CALCULATION ---
    def calc_wr(wins, total):
        return (wins / total * 100) if total > 0 else 0.0

    # --- PRINT GENERAL REPORT ---
    print("\n" + "="*75)
    print(f"{'🌍 GRAND TOTAL PORTFOLIO REPORT':^75}")
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
    
    # 1. LONG STATS
    long_wr = calc_wr(stats['long']['wins'], stats['long']['count'])
    print(f"{'LONG':<10} | {stats['long']['count']:<8} | %{long_wr:<14.2f} | {stats['long']['wins']}/{stats['long']['losses']:<12} | {stats['long']['pnl']:+.2f}")
    
    # 2. SHORT STATS
    short_wr = calc_wr(stats['short']['wins'], stats['short']['count'])
    print(f"{'SHORT':<10} | {stats['short']['count']:<8} | %{short_wr:<14.2f} | {stats['short']['wins']}/{stats['short']['losses']:<12} | {stats['short']['pnl']:+.2f}")
    
    print("-" * 75)
    
    # 3. TOTAL
    total_wr = calc_wr(stats['total']['wins'], stats['total']['count'])
    print(f"{'TOTAL':<10} | {stats['total']['count']:<8} | %{total_wr:<14.2f} | {stats['total']['wins']}/{stats['total']['losses']:<12} | {stats['total']['pnl']:+.2f}")
    print("="*75 + "\n")

if __name__ == "__main__":
    main()