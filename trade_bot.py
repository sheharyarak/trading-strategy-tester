# Trade Bot
from enum import Enum
import numpy as np
import typing
from datetime import datetime
import pandas as pd
import yfinance as yf

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# Cache Requests
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter

class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.limiter = Limiter(RequestRate(2, Duration.SECOND * 5))
            cls._instance.bucket_class = MemoryQueueBucket
            cls._instance.backend = SQLiteCache("yfinance.cache")

SESSION = CachedLimiterSession()

class TradeType(Enum):
    """Represents the type of trade (BUY or SELL)."""

    BUY = -1
    SELL = 1

    @classmethod
    def _missing_(cls, value):
        """Handles case-insensitive lookup of trade types by string."""
        if isinstance(value, str):
            return cls[value.upper()]
        return super()._missing_(value)

class Trade:
    """Represents a single stock trade."""

    def __init__(self, shares, price, timestamp, trade_type, expiration=-1):
        """Initializes a Trade object.

        Args:
            shares (int): Number of shares traded.
            price (float): Price per share.
            timestamp (datetime): Timestamp of the trade.
            trade_type (TradeType): Type of trade (BUY or SELL).
        """

        self.shares = shares
        self.price = price
        self.timestamp = timestamp
        self.trade_type = trade_type
        self.expiration = expiration

    @classmethod
    def buy(self, shares, price, timestamp, expiration=-1):
        """Creates a BUY trade.

        Args:
            shares (int): Number of shares to buy.
            price (float): Price per share.
            timestamp (datetime): Timestamp of the trade.

        Returns:
            Trade: A Trade object representing the BUY transaction.
        """

        return Trade(shares, price, timestamp, TradeType("BUY"), expiration)

    @classmethod
    def sell(self, shares, price, timestamp, expiration=-1):
        """Creates a SELL trade.

        Args:
            shares (int): Number of shares to sell.
            price (float): Price per share.
            timestamp (datetime): Timestamp of the trade.

        Returns:
            Trade: A Trade object representing the SELL transaction.
        """

        return Trade(shares, price, timestamp, TradeType("SELL"), expiration)

class TradeBot:
    """Simulates a simple trading bot."""

    def __init__(self, debug=False):
        """Initializes the TradeBot."""
        self.cash = 0
        self.shares = 0
        self.pending_trades = []
        self.completed_trades = []
        self.reset_history()
        self.debug = debug

    def reset_history(self):
        self.history = {
            "timestamp": list(),
            "cash": list(),
            "shares": list(),
            "net_worth": list(),
        }

    def log(self, timestamp, price):
        self.history["timestamp"].append(timestamp)
        self.history["cash"].append(self.cash)
        self.history["shares"].append(self.shares * price)
        self.history["net_worth"].append(self.cash + (self.shares * price))

    def deposit(self, cash):
        """Deposits cash into the TradeBot's account.

        Args:
            cash (float): Amount of cash to deposit.
        """
        self.cash += cash

    def trade(self, trade):
        """Executes a trade.

        Args:
            trade (Trade): The Trade object to execute.
        """
        if trade.trade_type == TradeType("BUY"):
            cash_diff = (trade.shares * trade.price) * trade.trade_type.value
            if self.cash >= cash_diff:
                self.debug and print(f"BUY: {trade.shares} @ {trade.price} = {cash_diff}; CASH: {self.cash} + {cash_diff} = {self.cash + cash_diff}")
                self.cash += cash_diff
            else:
                return # Invalid Trade
        elif trade.trade_type == TradeType("SELL"):
            share_diff = trade.shares * (trade.trade_type.value * -1)
            if self.shares >= share_diff:
                self.debug and print(f"SELL: {trade.shares} @ {trade.price}; SHARES: {self.shares} + {share_diff} = {self.shares + share_diff}")
                self.shares += share_diff
            else:
                return # Invalid Trade
        self.pending_trades.append(trade)

    def expire(self, trade):
        if trade.trade_type == TradeType("BUY"):
            cash_diff = (trade.shares * trade.price) * (trade.trade_type.value * -1)
            self.debug and print(f"[EXPIRED] BUY: {trade.shares} @ {trade.price} = {cash_diff}; CASH: {self.cash} + {cash_diff} = {self.cash + cash_diff}")
            self.cash += cash_diff
        elif trade.trade_type == TradeType("SELL"):
            share_diff = trade.shares * trade.trade_type.value
            self.debug and print(f"[EXPIRED] SELL: {trade.shares} @ {trade.price}; SHARES: {self.shares} + {share_diff} = {self.shares + share_diff}")
            self.shares += share_diff

    def settle(self, trade):
        if trade.trade_type == TradeType("BUY"):
            share_diff = trade.shares * (trade.trade_type.value * -1)
            self.debug and print(f"[SETTLED] BUY: {trade.shares} @ {trade.price}; SHARES: {self.shares} + {share_diff} = {self.shares + share_diff}")
            self.shares += share_diff
        elif trade.trade_type == TradeType("SELL"):
            cash_diff = (trade.shares * trade.price) * trade.trade_type.value
            self.debug and print(f"[SETTLED] BUY: {trade.shares} @ {trade.price} = {cash_diff}; CASH: {self.cash} + {cash_diff} = {self.cash + cash_diff}")
            self.cash += (trade.shares * trade.price) * trade.trade_type.value
        self.completed_trades.append(trade)

    def settle_pending_trades(self, price):
        indices_to_delete = []
        for i, p_trade in enumerate(self.pending_trades):
            if p_trade.price < price:
                self.settle(p_trade)
                indices_to_delete.append(i)
        for i in reversed(indices_to_delete):
            del self.pending_trades[i]

    def expire_pending_trades(self):
        indices_to_delete = []
        for i, p_trade in enumerate(self.pending_trades):
            if p_trade.expiration == 0:
                self.expire(p_trade)
                indices_to_delete.append(i)
            else:
                p_trade.expiration -= 1
        for i in reversed(indices_to_delete):
            del self.pending_trades[i]

    def play(self, strategy, ticker_name, session=SESSION, **history_kwargs):
        """Executes a trading strategy using historical data.

        Args:
            strategy (Strategy): The trading strategy to use.
            ticker_name (str): Name of the stock ticker.
            session:  A yfinance session object.
            **history_kwargs: Arguments to pass to yfinance for retrieving history.
        """
        ticker = yf.Ticker(ticker_name, session=session)
        ticker_history = ticker.history(**history_kwargs).reset_index()
        ticker_history.columns = ticker_history.columns.str.lower()
        ticker_history.rename(columns={'datetime': 'timestamp', 'stock splits': 'stock_splits'}, inplace=True)
        self.reset_history()
        start_price = ticker_history.head(1).iloc[0]["open"]
        print(f"Starting Net Worth: Cash: {self.cash}, Shares: {self.shares} @ {start_price} = {self.cash + self.shares * start_price}")
        for index, row in ticker_history.iterrows():
            self.expire_pending_trades()
            self.settle_pending_trades(row["open"])
            self.settle_pending_trades(row["close"])
            kwargs = dict(cash=self.cash, shares=self.shares, ticker_name=ticker_name, **history_kwargs, **row)
            trade = strategy.play(**kwargs)
            if trade:
                self.trade(trade)
            self.log(row["timestamp"], row["close"])
        end_price = ticker_history.tail(1).iloc[0]["close"]
        print(f"Ending Net Worth: Cash: {self.cash}, Shares: {self.shares} @ {start_price} = {self.cash + self.shares * end_price}")
        net_worth_change = ((self.cash + self.shares * end_price) - (self.cash + self.shares * start_price)) / (self.cash + self.shares * start_price)
        stock_change = (end_price - start_price) / start_price
        print(f"Your Net Worth changed {net_worth_change * 100: .2f}%.")
        print(f"The Stock Prick changed {stock_change * 100: .2f}%.")

    def plot_history(self, figsize=(50, 10)):
        """
        Plots cash, shares, and net worth over time using Seaborn.

        Args:
            df (pd.DataFrame): Dataframe containing columns 'timestamp', 'cash', 'shares', and 'net_worth'.
            figsize (tuple, optional): Desired figure size. Defaults to (12, 6).
        """
        df = pd.DataFrame(self.history)
        # Ensure the DataFrame has the required columns
        required_cols = ['timestamp', 'cash', 'shares', 'net_worth']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("DataFrame must contain columns: {}".format(required_cols))

        # Prepare for long-format plotting with Seaborn
        df = df.reset_index()  # Reset timestamp to a normal column
        df_long = pd.melt(df, id_vars='timestamp', value_vars=['cash', 'shares', 'net_worth'], var_name='Metric', value_name='Value')

        # Create the Seaborn plot
        sns.relplot(
            data=df_long,
            x='timestamp',
            y='Value',
            hue='Metric',
            kind='line',
            height=figsize[1],
            aspect=figsize[0]/figsize[1]
        )

        # Formatting: Control x-axis ticks
        plt.xticks(rotation=90)  # Keep the rotation
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatic, spaced ticks
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M')) # Full timestamp format
        plt.legend(title='')
        plt.show()
