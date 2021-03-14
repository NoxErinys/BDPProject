from typing import List, Dict
from models import Visualizer, Trader, Prediction
from datetime import datetime


class MatPlotLibVisualizer(Visualizer):

    def plot_predictions(self, data: List[Prediction]):
        pass
        # VLAD

    def update_traders_plot(self, timestamp: datetime, traders: List[Trader], current_prices: Dict[str, float]):
        pass
        # VLAD
