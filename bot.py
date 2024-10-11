import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Sequence
from warnings import filterwarnings

import numpy as np
import pandas as pd
import tomllib
import yfinance as yf
from loguru import logger
from tqdm import tqdm


class TradingBot:
    def __init__(self, symbols: Sequence[str], interval: str = "1h") -> None:
        filterwarnings("ignore")

        with open("parameters.toml", "r") as f:
            self.parameters = tomllib.loads(f.read())

        self.symbols = symbols
        self.interval = interval

        # Determine the maximum lookback needed for indicators
        max_lookback = max(
            self.parameters.get("lookback", 20),  # For trend breakout and Fibonacci
            20,  # For Bollinger Bands
            14,  # For RSI_14
        )

        # Calculate the number of periods per day based on interval
        interval_minutes = int(interval.strip("m").strip("h"))
        trading_minutes_per_day = 6.5 * 60  # US market open for 6.5 hours per day
        periods_per_day = int(trading_minutes_per_day / interval_minutes)

        # Calculate the number of days needed
        days_needed = (
            int(np.ceil((max_lookback + 50) / periods_per_day)) + 1
        )  # Adding extra periods and a buffer day

        self.start_date = datetime.now(timezone.utc) - timedelta(days=days_needed)
        self.end_date = datetime.now(timezone.utc)

        self.data = {}

    def download_data(self):
        for symbol in tqdm(self.symbols, desc="Downloading data..."):
            df = yf.download(
                symbol,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                prepost=False,
                auto_adjust=True,
                progress=False,
                repair=True,
            )
            df.dropna(inplace=True)
            self.data[symbol] = df

    def calculate_indicators(self, symbol: str):
        df = self.data[symbol]

        df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

        df["RSI_9"] = self.calculate_rsi(df["Close"], 9)
        df["RSI_14"] = self.calculate_rsi(df["Close"], 14)

        df["BB_Middle"] = df["Close"].rolling(window=20).mean()
        df["BB_Upper"] = df["BB_Middle"] + 2 * df["Close"].rolling(window=20).std()
        df["BB_Lower"] = df["BB_Middle"] - 2 * df["Close"].rolling(window=20).std()

        self.identify_support_resistance(symbol)

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

            if signal:
                try:
                    timestamp = current.name.tz_localize("UTC").tz_convert(
                        "America/Sao_Paulo"
                    )
                except TypeError:
                    timestamp = current.name.tz_convert("America/Sao_Paulo")
                current_time = pd.Timestamp.now(tz="America/Sao_Paulo")
                time_difference = current_time - timestamp

                if time_difference <= pd.Timedelta(minutes=120):
                    signal["Date"] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    signals.append(signal)

        return signals

    def dispatch(self, content: Sequence[str]):
        recipients = self.parameters["dispatcher"]["recipients"]

        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        smtp_username = self.parameters["dispatcher"]["sender"]
        smtp_password = self.parameters["dispatcher"]["password"]

        sender_email = self.parameters["dispatcher"]["sender"]
        subject = f"BBG Challenge 2024 Signals {datetime.now()}"

        message_body = "\n".join(content)

        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject

        msg.attach(MIMEText(message_body, "plain"))

        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.sendmail(sender_email, recipients, msg.as_string())
            server.quit()
            print("Email sent successfully.")
        except Exception as e:
            print("Failed to send email:", str(e))

    def execute(self):
        self.download_data()
        all_signals = []

        for symbol in tqdm(self.symbols, desc="Calculating Indicators..."):
            self.calculate_indicators(symbol)
            signals = self.generate_signals(symbol)
            for signal in signals:
                out = f"Timestamp: {signal['Date']}, Symbol: {symbol}, Action: {signal['Action']}, Price: {signal['Price']}\n"
                if "Stop_Loss" in signal:
                    out += f'Stop Loss set at: {signal["Stop_Loss"]} \n'

                logger.info(out)
                print(out)
                all_signals.append(out)

        if all_signals:
            all_signals.sort(key=lambda x: x.split(",")[0])
            self.dispatch(all_signals)
