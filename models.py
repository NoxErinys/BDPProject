from typing import List, Dict
from datetime import datetime
from abc import ABC, abstractmethod


class DataPoint:

    def __init__(self, open_price: float, close_price: float, highest_price: float,
                 lowest_price: float, volume: float, timestamp: datetime):
        self.open_price = open_price
        self.close_price = close_price
        self.highest_price = highest_price
        self.lowest_price = lowest_price
        self.volume = volume
        self.timestamp = timestamp


class Prediction:

    def __init__(self, stock: str, current_timestamp: datetime, predicted_timestamp: datetime,
                 current_price: float, predicted_price: float):
        self.stock = stock
        self.current_timestamp = current_timestamp
        self.predicted_timestamp = predicted_timestamp
        self.current_price = current_price
        self.predicted_price = predicted_price


class DataCollector(ABC):

    def __init__(self, start_date: datetime, interval_in_seconds: int):
        self.start_date = start_date
        self.interval_in_seconds = interval_in_seconds

    @abstractmethod
    def get_top_stocks(self, number_of_stocks=100) -> List[str]:
        pass

    @abstractmethod
    def get_historical_data(self, stock: str) -> List[DataPoint]:
        pass

    @abstractmethod
    def get_latest_data_point(self, stocks: List[str]) -> Dict[str, DataPoint]:
        pass


class Predictor(ABC):

    def __init__(self, stock: str):
        self.stock = stock

    @abstractmethod
    def train(self, data: List[DataPoint]):
        pass

    @abstractmethod
    def predict_next_price(self, current_data_point: DataPoint) -> Prediction:
        pass

    @abstractmethod
    def update_model(self, current_data_point: DataPoint):
        pass


class StockPurchase:

    def __init__(self, amount: float, price: float, timestamp: datetime):
        self.amount = amount
        self.price = price
        self.timestamp = timestamp


class StockBalanceSheet:

    def __init__(self, name: str, purchases: List[StockPurchase]):
        self.name = name
        self.purchases = purchases

    def get_available_amount(self):
        return sum([s.amount for s in self.purchases])

    def get_unrealized_profit(self, current_price: float):
        owned_value = sum([s.amount * s.price for s in self.purchases])
        current_value = current_price * self.get_available_amount()

        return current_value - owned_value


class Trader(ABC):

    def __init__(self, name: str, balance: float):
        self.name = name
        self.initial_balance = balance
        self.balance = balance
        self.balance_sheet: Dict[str, StockBalanceSheet] = {}

    @abstractmethod
    def trade(self, predictions: List[Prediction]):
        pass

    def get_current_net_worth(self, stock_prices: Dict[str, float]):
        return self.balance +\
               sum([s.get_unrealized_profit(stock_prices[s.name]) for s in self.balance_sheet.values()])

    def get_net_worth(self, timestamp: datetime, stock_prices: Dict[str, float]):
        """
        INITIAL_BALANCE - BUYS UP UNTIL timestamp + UNRELEASED PROFIT UP UNTIL timestamp
        TODO: FINISH UP
        """
        return self.balance


class Visualizer(ABC):

    @abstractmethod
    def plot_predictions(self, data: List[Prediction]):
        pass

    @abstractmethod
    def update_traders_plot(self, timestamp: datetime, traders: List[Trader], current_prices: Dict[str, float]):
        pass


class Bot:

    def __init__(self, trader: Trader, predictiors: Dict[str, Predictor]):
        self.trader = trader
        self.predictors = predictiors
