from typing import List
from datetime import datetime, timedelta
from models import Trader, DataPoint, Prediction


class SimpleTrader(Trader):

    def __init__(self, name: str, balance: float):
        super().__init__(name, balance)

    def trade(self, predictions: List[Prediction]):
        for prediction in predictions:

            stocks_to_trade = float((self.balance / len(predictions)) / prediction.current_price / 1.2)
            last_transaction = self.get_last_transaction_for_stock(prediction.stock)

            price_difference = prediction.predicted_price / prediction.current_price
            price_difference_to_last_transaction = 0 if last_transaction is None \
                else prediction.predicted_price / last_transaction.price

            if 0.997 <= price_difference <= 1.003 and 0.993 <= price_difference_to_last_transaction <= 1.008:
                continue

            if prediction.predicted_price > prediction.current_price and \
                    self.balance > prediction.current_price * stocks_to_trade and \
                    (last_transaction is None or
                     (last_transaction.amount < 0 and last_transaction.price > prediction.current_price)):

                self.buy(prediction.stock, stocks_to_trade, prediction.current_price, prediction.current_timestamp,
                         prediction.predicted_price)

            elif prediction.predicted_price < prediction.current_price and last_transaction is not None and\
                    self.get_currently_available_amount_for_stock(prediction.stock) >= stocks_to_trade and \
                    last_transaction.amount > 0 and last_transaction.price < prediction.current_price:

                self.sell(prediction.stock, stocks_to_trade, prediction.current_price, prediction.current_timestamp,
                          prediction.predicted_price)

    def get_last_transaction_for_stock(self, stock):
        return self.balance_sheet[stock].purchases[-1] if stock in self.balance_sheet else None


def test():
    pass


if __name__ == '__main__':
    # If this file is executed directly, the below examples will be run and tested:
    test()
