import typing
import pandas as pd
from trade_bot import Trade

class Strategy:
    """Base class for trading strategies."""

    def play(
        self,
        *,
        cash: float,
        shares: int,
        timestamp: pd._libs.tslibs.timestamps.Timestamp,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: int,
        dividends: float,
        stock_splits: float,
        **kwargs
    ) -> typing.Optional[Trade]:
        """Implements the trading strategy logic based on historical and current market data.

        This method is the core of your trading strategy. It receives various data points.
        and the current state of the TradeBot to make trading decisions.

        Args:
            ticker_name (str): The name of the stock ticker symbol.
            cash (float): The current cash balance of the TradeBot.
            shares (int): The current number of shares held by the TradeBot for this stock.
            timestamp (pd._libs.tslibs.timestamps.Timestamp): The datetime of the current data point.
            open_ (float): The opening price of the stock on that date.
            high (float): The highest price of the stock on that date.
            low (float): The lowest price of the stock on that date.
            close (float): The closing price of the stock on that date.
            volume (int): The trading volume for that date.
            dividends (float): Any dividends distributed for that date (usually 0).
            stock_split (float): Any stock split factor for that date (usually 1).
            **kwargs (dict): Optional keyword arguments that may be passed by the TradeBot.play method.

        Returns:
            typing.Optional[Trade]:
                - Returns a `Trade` object if the strategy decides to execute a trade (BUY or SELL).
                - Returns `None` if the strategy chooses not to trade at this point.

        **Note:** Subclasses implementing specific trading strategies must define their own logic
        within this method to analyze the provided data and make trading decisions.
        """

        raise NotImplementedError("Subclass must implement the play method")