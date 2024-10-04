import time
import tomllib
from datetime import datetime, timedelta, timezone
from typing import Sequence

import numpy as np
import pandas as pd
import schedule
import yfinance as yf


class TradingBot:
    def __init__(self, symbols: Sequence[str], interval: str = "15m") -> None:
        with open("parameters.toml", "r") as f:
            self.parameters = tomllib.loads(f.read())

        self.symbols = symbols
        self.interval = interval

        # Determine the maximum lookback needed for indicators
        max_lookback = max(
            200,  # For EMA_200
            self.parameters.get("lookback", 20),  # For trend breakout and Fibonacci
            20,  # For Bollinger Bands
            14,  # For RSI_14
        )

        # Calculate the number of periods per day based on interval
        interval_minutes = int(interval.strip("m"))
        trading_minutes_per_day = 6.5 * 60  # US market open for 6.5 hours per day
        periods_per_day = int(trading_minutes_per_day / interval_minutes)

        # Calculate the number of days needed
        days_needed = (
            int(np.ceil((max_lookback + 50) / periods_per_day)) + 1
        )  # Adding extra periods and a buffer day

        start_date = datetime.now(timezone.utc) - timedelta(days=days_needed)
        end_date = datetime.now(timezone.utc)

        self.data = {}
        for symbol in symbols:
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                prepost=False,
                auto_adjust=True,
                progress=False,
            )
            df.dropna(inplace=True)
            self.data[symbol] = df

    def calculate_indicators(self, symbol: str):
        df = self.data[symbol]

        df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
        df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

        df["RSI_9"] = self.calculate_rsi(df["Close"], 9)
        df["RSI_14"] = self.calculate_rsi(df["Close"], 14)

        df["BB_Middle"] = df["Close"].rolling(window=20).mean()
        df["BB_Upper"] = df["BB_Middle"] + 2 * df["Close"].rolling(window=20).std()
        df["BB_Lower"] = df["BB_Middle"] - 2 * df["Close"].rolling(window=20).std()

        self.identify_support_resistance(symbol)

        self.detect_fibonacci_retracements(symbol)

        self.detect_trend_breakouts(symbol)

    def calculate_rsi(self, series: pd.Series, period: int):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def identify_support_resistance(self, symbol: str):
        df = self.data[symbol]
        df["Support"] = df["Low"].rolling(window=20).min()
        df["Resistance"] = df["High"].rolling(window=20).max()

    def detect_fibonacci_retracements(self, symbol: str):
        df = self.data[symbol]
        lookback = self.parameters.get("lookback", 20)
        fibonacci_levels = self.parameters.get("fibonacci", {}).get(
            "levels", [0.236, 0.382, 0.5, 0.618, 0.786]
        )

        for level in fibonacci_levels:
            df[f"Fib_{int(level*100)}"] = np.nan

        for i in range(lookback, len(df)):
            window = df.iloc[i - lookback : i]
            max_price = window["High"].max()
            min_price = window["Low"].min()
            diff = max_price - min_price
            for level in fibonacci_levels:
                fib_level = max_price - level * diff
                df.at[df.index[i], f"Fib_{int(level*100)}"] = fib_level

    def detect_trend_breakouts(self, symbol: str):
        df = self.data[symbol]
        lookback = self.parameters.get("lookback", 20)

        df["Trend_Breakout_Up"] = False
        df["Trend_Breakout_Down"] = False

        for i in range(lookback, len(df)):
            window = df.iloc[i - lookback : i]
            max_high = window["High"].max()
            min_low = window["Low"].min()
            current_close = df["Close"].iloc[i]

            if current_close > max_high:
                df.at[df.index[i], "Trend_Breakout_Up"] = True
            elif current_close < min_low:
                df.at[df.index[i], "Trend_Breakout_Down"] = True

    def generate_signals(self, symbol: str):
        df = self.data[symbol]
        signals = []

        for i in range(1, len(df)):
            signal = {}
            current = df.iloc[i]
            previous = df.iloc[i - 1]

            if (
                current["Close"] > current["EMA_9"]
                and previous["Close"] <= previous["EMA_9"]
            ):
                signal["Action"] = "Buy"
                signal["Price"] = current["Close"]
                signal["Stop_Loss"] = df["Low"].iloc[i - 1]

            elif current["Close"] >= current["Resistance"]:
                signal["Action"] = "Sell"
                signal["Price"] = current["Close"]

            elif current.get("Trend_Breakout_Up", False):
                signal["Action"] = "Buy (Breakout Up)"
                signal["Price"] = current["Close"]
            elif current.get("Trend_Breakout_Down", False):
                signal["Action"] = "Sell (Breakout Down)"
                signal["Price"] = current["Close"]

            if signal and current.name.date() == pd.Timestamp.now(tz="UTC").date():
                timestamp = current.name.tz_localize("UTC").tz_convert(
                    "America/Sao_Paulo"
                )
                signal["Date"] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                signals.append(signal)

        return signals

    def execute(self):
        for symbol in self.symbols:
            self.calculate_indicators(symbol)
            signals = self.generate_signals(symbol)
            for signal in signals:
                print(
                    f"Timestamp: {signal['Date']}, Symbol: {symbol}, Action: {signal['Action']}, Price: {signal['Price']}\n{f'Stop Loss set at: {signal['Stop_Loss']}' if 'Stop_Loss' in signal else ''}\n"
                )

    def run(self):
        def job():
            current_time = datetime.now()
            print(f"Running at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.execute()

        current_time = datetime.now()
        minutes_to_next = 15 - (current_time.minute % 15)
        next_run = current_time + timedelta(minutes=minutes_to_next)

        schedule.every().day.at(next_run.strftime("%H:%M")).do(job)

        schedule.every(15).minutes.do(job)

        while True:
            schedule.run_pending()
            time.sleep(1)
