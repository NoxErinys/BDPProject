from typing import List

from models import Trader, DataPoint, Prediction


class SafeTrader(Trader):

    def __init__(self, name: str, balance: float):
        super().__init__(name, balance)

    def trade(self, predictions: List[Prediction]):
        pass
        # TOGETHER
