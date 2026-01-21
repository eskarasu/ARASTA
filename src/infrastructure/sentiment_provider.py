import requests
import json
import os
import time
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
from src.config.settings import settings
from src.infrastructure.logger import logger

class SentimentProvider:
    """
    Provider managing Fear & Greed Index data.
    Fetches historical data, stores it, and provides data for future prediction.
    """
    def __init__(self):
        self.api_url = "https://api.alternative.me/fng/"
        self.file_path = settings.SENTIMENT_HISTORY_FILE
        self.data: List[Dict] = []
        self.lookup_map: Dict[str, int] = {}
        self._load_data()

    def _load_data(self):
        """Loads data from local file, otherwise fetches from API"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    self.data = json.load(f)
                
                # Update if data is too old (more than 1 day)
                if self.data:
                    last_ts = int(self.data[0]['timestamp'])
                    if time.time() - last_ts > 86400:
                        self.fetch_history()
                self._build_lookup_map()
            except Exception as e:
                logger.error(f"Error reading sentiment data: {e}")
                self.fetch_history()
        else:
            self.fetch_history()

    def fetch_history(self):
        """Downloads all historical data from API"""
        try:
            # limit=0 fetches all history
            resp = requests.get(f"{self.api_url}?limit=0&format=json", timeout=10)
            if resp.status_code == 200:
                raw_data = resp.json().get('data', [])
                # Clean and format data
                self.data = []
                for item in raw_data:
                    self.data.append({
                        'value': int(item['value']),
                        'value_classification': item['value_classification'],
                        'timestamp': int(item['timestamp']),
                        'date': datetime.fromtimestamp(int(item['timestamp'])).strftime('%Y-%m-%d')
                    })
                
                # Save
                with open(self.file_path, 'w') as f:
                    json.dump(self.data, f)
                logger.info(f"🧠 Sentiment History Updated: {len(self.data)} days")
                self._build_lookup_map()
            else:
                logger.warning("Sentiment API did not respond.")
        except Exception as e:
            logger.error(f"Sentiment download error: {e}")

    def _build_lookup_map(self):
        """Builds index map for fast date lookup"""
        self.lookup_map = {item['date']: i for i, item in enumerate(self.data)}

    def get_recent_sentiment(self, days: int = 7) -> pd.DataFrame:
        """Returns last N days data as DataFrame"""
        if not self.data:
            return pd.DataFrame()
        
        # Data comes from API "Newest to Oldest" (Index 0 is newest)
        recent = self.data[:days]
        # Sort "Oldest to Newest" for analysis
        df = pd.DataFrame(recent).sort_values('timestamp', ascending=True)
        return df

    def predict_next_move(self) -> Dict:
        """
        Predicts next move using simple momentum analysis.
        """
        df = self.get_recent_sentiment(days=14)
        if df.empty or len(df) < 5:
            return {'trend': 'NEUTRAL', 'predicted_value': 50, 'confidence': 0.0}

        values = df['value'].values
        
        # 1. Momentum (Change of last 3 days)
        momentum = values[-1] - values[-3]
        
        # 2. Simple Linear Regression (Slope)
        x = range(len(values))
        slope = 0
        try:
            slope, _ = np.polyfit(x, values, 1)
        except:
            slope = 0

        # 3. Prediction
        current_val = values[-1]
        predicted_val = current_val + (slope * 2) # Roughly predict 2 days ahead
        predicted_val = max(0, min(100, predicted_val))

        trend = 'NEUTRAL'
        if slope > 1.5: trend = 'RAPID_RECOVERY' # Rapid recovery (Strong Buy)
        elif slope > 0.5: trend = 'SLOW_RECOVERY'
        elif slope < -1.5: trend = 'RAPID_FEAR' # Panic sell (Strong Sell/Wait)
        elif slope < -0.5: trend = 'SLOW_FEAR'

        return {
            'trend': trend,
            'current_value': current_val,
            'predicted_value': predicted_val,
            'slope': slope,
            'is_reversal': (values[-1] > values[-2] and values[-2] < values[-3]) # Is it a reversal from bottom?
        }

    def get_sentiment_at_date(self, target_date: datetime) -> Optional[Dict]:
        """
        For Backtest: Calculates sentiment value and trend at a specific date.
        """
        if not self.data: return None
        
        date_str = target_date.strftime('%Y-%m-%d')
        idx = self.lookup_map.get(date_str)
        
        if idx is None:
            return None
            
        # Data for that day
        current_item = self.data[idx]
        
        # Get previous 14 days (Data is Newest->Oldest so idx increases)
        # E.g.: idx=100 (Today), idx=101 (Yesterday)... idx+14 (14 days ago)
        history_slice = self.data[idx : idx+14]
        
        # Analysis (Sort Oldest to Newest)
        df = pd.DataFrame(history_slice).sort_values('timestamp', ascending=True)
        
        trend = 'NEUTRAL'
        slope = 0.0
        
        if len(df) >= 5:
            values = df['value'].values
            try:
                x = range(len(values))
                slope, _ = np.polyfit(x, values, 1)
                
                if slope > 1.5: trend = 'RAPID_RECOVERY'
                elif slope > 0.5: trend = 'SLOW_RECOVERY'
                elif slope < -1.5: trend = 'RAPID_FEAR'
                elif slope < -0.5: trend = 'SLOW_FEAR'
            except: pass
            
        return {
            'value': current_item['value'],
            'trend': trend,
            'slope': slope
        }
