# ARASTA: Adaptive Regime And Sentiment Trading Agent
<img width="500" height="500" alt="ARASTA2" src="https://github.com/user-attachments/assets/7148c4a3-dd75-4c7e-82f7-3bf4e9d6e344" />


This project is an automated trading bot and simulation (backtest) engine running on the Binance exchange, making trading decisions by combining advanced technical analysis indicators and market sentiment data.

## 🌟 Key Differentiators (Why this bot?)

Unlike standard bots that rely on simple indicator crossovers, this system implements a **"Market Regime First"** approach:

*   **🧠 Macro-Trend Awareness:** It never trades against the major BTC trend. If BTC is bearish (Price < EMA50 < EMA200), it switches to Short-only or defensive mode.
*   **❤️ Sentiment Integration:** Incorporates the **Fear & Greed Index** into its decision logic. It knows when to "buy the fear" or "sell the greed".
*   **🛡️ Dynamic Risk Management:** Uses **ATR-based Trailing Stops** that automatically tighten as profits grow to lock in gains.
*   **💯 Scoring System:** Instead of binary signals (Buy/Sell), it calculates a **Composite Score** based on multiple factors (RSI, MACD, Volume, Bollinger Bands) to filter out weak signals.

## 🚀 Features

*   **Smart Market Analysis:** Generates entry/exit signals using RSI, MACD, Bollinger Bands, and Volume data.
*   **Backtest Engine:** Tests strategies on historical data, providing PnL (Profit/Loss) and Win Rate reports.
*   **Database Integration:** Stores all trade history and bot logs in an SQLite database.

## 🛠 Installation

1.  **Clone the Project:**
    ```bash
    git clone https://github.com/eskarasu/ARASTA.git
    cd ARASTA
    ```

2.  **Install Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Installing `TA-Lib` may require additional steps depending on your operating system.*

3.  **Configuration (.env):**
    Create a copy of the example settings file and rename it to `.env`:
    ```bash
    cp .env.example .env
    ```
    Then open the `.env` file and enter your Binance API keys and preferred settings.

## 📈 Usage

### 🚀 Running the Bot (Live/Simulation)
To start the bot in the mode defined in your `.env` file (Live or Simulation):

```bash
python src/main.py
```

### 🧪 Running Backtest

**Option 1: Auto-Pattern Matching**
To find a historical period similar to the current market conditions and test on that:

```bash
python run_backtester.py
```

**Option 2: Exact Date Range**
Run the following command to test the strategy over a specific date range:

```bash
python run_backtester_exact_date.py
```

This script performs the following:
1. Downloads historical data from Binance for specified symbols.
2. Simulates the market regime (Trend and Sentiment).
3. Executes trading transactions and prints the performance report to the console.

## ⚠️ Legal Disclaimer

This software is developed strictly for educational and testing purposes. Cryptocurrency markets involve high risk. The developer shall not be held responsible for any financial losses resulting from the use of this software. This is not investment advice.

---
