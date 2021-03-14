from typing import List

from models import Prediction, DataPoint, Predictor


class RNNPredictor(Predictor):

    def __init__(self, stock: str):
        super().__init__(stock)

    def train(self, data: List[DataPoint]):
        pass
        # BOTH, INDIVIDUALLY

    def predict_next_price(self, current_data_point: DataPoint) -> Prediction:
        pass
        # BOTH, INDIVIDUALLY

    def update_model(self, actual_data_point: DataPoint):
        pass
        # BOTH, INDIVIDUALLY


class ImpairedRNNPredictor(Predictor):

    def __init__(self, stock: str):
        super().__init__(stock)

    def train(self, data: List[DataPoint]):
        pass
        # BOTH, INDIVIDUALLY

    def predict_next_price(self, current_data_point: DataPoint) -> Prediction:
        pass
        # BOTH, INDIVIDUALLY

    def update_model(self, actual_data_point: DataPoint):
        pass
        # BOTH, INDIVIDUALLY

