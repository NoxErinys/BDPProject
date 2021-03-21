from typing import List, Dict
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import jsonpickle
import pathlib


class DataPoint:

    def __init__(self, open_price: float, close_price: float, highest_price: float,
                 lowest_price: float, volume: float, timestamp: datetime):
        self.open_price = open_price
        self.close_price = close_price
        self.highest_price = highest_price
        self.lowest_price = lowest_price
        self.volume = volume
        self.timestamp = timestamp

    def __repr__(self):
        return f"{self.close_price} at {str(self.timestamp)}"


class Prediction:

    def __init__(self, stock: str, current_timestamp: datetime, predicted_timestamp: datetime,
                 current_price: float, predicted_price: float):
        self.stock = stock
        self.current_timestamp = current_timestamp
        self.predicted_timestamp = predicted_timestamp
        self.current_price = current_price
        self.predicted_price = predicted_price

    def __repr__(self):
        return f"{self.stock}: {self.current_price} at {self.current_timestamp} | {self.predicted_price} at {self.predicted_timestamp}"


class DataCollector(ABC):

    def __init__(self, interval_in_seconds: int):
        self.interval_in_seconds = interval_in_seconds

    @abstractmethod
    def get_top_stocks(self, current_time: datetime,  number_of_stocks=10) -> List[str]:
        pass

    @abstractmethod
    def get_historical_data(self, stock: str, current_time: datetime, number_of_days: int = 10) -> List[DataPoint]:
        pass

    @abstractmethod
    def get_latest_data_point(self, stocks: List[str], current_time: datetime) -> Dict[str, DataPoint]:
        pass


class Predictor(ABC):

    def __init__(self, stock: str, data_interval: timedelta, window: int=100, epochs: int=32):
        self.stock = stock
        self.data_interval = data_interval
        self.window = window
        self.epochs = epochs

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

    def __init__(self, stock_name: str, purchases: List[StockPurchase]):
        self.stock_name = stock_name
        self.purchases = purchases
        self._available_stocks = 0

    def get_current_available_amount(self):
        return self._available_stocks

    def get_current_unrealized_profit(self, current_price: float):
        return self._available_stocks * current_price

    def get_available_stocks_average_price(self):
        stock_purchases = [p for p in self.purchases if p.amount > 0]

        return sum([p.amount * p.price for p in stock_purchases]) /\
               sum([p.amount for p in stock_purchases])

    def get_available_amount_at_time(self, timestamp: datetime):
        return sum([p.amount for p in self.purchases if p.timestamp <= timestamp])

    def get_unrealized_profit_at_time(self, price_at_timestamp: float, timestamp: datetime):
        return self.get_available_amount_at_time(timestamp) * price_at_timestamp

    def get_total_value_at_time(self, timestamp: datetime):
        return sum([p.amount * p.price * -1 for p in self.purchases if p.timestamp <= timestamp])

    def buy(self, amount: float, price: float, timestamp: datetime):
        self._available_stocks += amount
        self.purchases.append(StockPurchase(amount, price, timestamp))

        return amount * price

    def sell(self, amount: float, price: float, timestamp: datetime):
        if amount > self._available_stocks:
            raise Exception("Tried to sell more than owned")

        self._available_stocks -= amount
        self.purchases.append(StockPurchase(-amount, price, timestamp))

        return amount * price


class Trader(ABC):

    def __init__(self, name: str, balance: float):
        self.name = name
        self.initial_balance = balance
        self.balance = balance
        self.balance_sheet: Dict[str, StockBalanceSheet] = {}
        self.is_loaded_from_file = False

    @abstractmethod
    def trade(self, predictions: List[Prediction]):
        pass

    def buy(self, stock: str, amount: float, price: float, timestamp: datetime, predicted_price=None):
        if self.balance - amount * price < 0:
            raise Exception("Can't buy more than the available balance")

        if stock not in self.balance_sheet:
            self.balance_sheet[stock] = StockBalanceSheet(stock, [])

        self.balance -= self.balance_sheet[stock].buy(amount, price, timestamp)
        logging.info("{trader_name} BOUGHT {amount} of {stock} at {price} because of prediction {prediction}. " +
                     "Balance now: {balance}", trader_name=self.name, stock=stock, amount=amount, price=price,
                                               prediction=str(predicted_price), balance=self.balance)

    def sell(self, stock: str, amount: float, price: float, timestamp: datetime, predicted_price=None):
        if stock not in self.balance_sheet:
            raise Exception("Can't sell stock because it's not owned")

        self.balance += self.balance_sheet[stock].sell(amount, price, timestamp)
        logging.info("{trader_name} SOLD {amount} of {stock} at {price} because of prediction {prediction}. " +
                     "Balance now: {balance}", trader_name=self.name, stock=stock, amount=amount, price=price,
                                               prediction=str(predicted_price), balance=self.balance)

    def get_currently_available_amount_for_stock(self, stock: str):
        if stock not in self.balance_sheet:
            return 0

        return self.balance_sheet[stock].get_current_available_amount()

    def get_current_net_worth(self, stock_prices: Dict[str, float]):
        return self.balance +\
            sum([s.get_current_unrealized_profit(stock_prices[s.stock_name]) for s in self.balance_sheet.values()])

    def get_net_worth_at_time(self, timestamp: datetime, stock_prices: Dict[str, float]):
        balance = self.initial_balance + sum([s.get_total_value_at_time(timestamp)
                                              for s in self.balance_sheet.values()])

        return balance + \
            sum([s.get_unrealized_profit_at_time(stock_prices[s.stock_name], timestamp)
                 for s in self.balance_sheet.values()])

    def save_data(self, file_path):
        path = pathlib.Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as out_file:
            out_file.write(jsonpickle.encode(self, indent=4))

    @staticmethod
    def load_data(file_path):
        with open(file_path, 'r') as out_file:
            data = out_file.read()
            trader_data = jsonpickle.decode(data)
            trader_data.is_loaded_from_file = True

            return trader_data


class Visualizer(ABC):

    @abstractmethod
    def update_predictions_plot(self, data: List[Prediction]):
        pass

    @abstractmethod
    def update_traders_plot(self, timestamp: datetime, traders: List[Trader], current_prices: Dict[str, float]):
        pass


class Bot:

    def __init__(self, trader: Trader, predictors: Dict[str, Predictor]):
        self.trader = trader
        self.predictors = predictors
