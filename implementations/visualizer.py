from typing import List, Dict
from models import Visualizer, Trader, Prediction
from datetime import datetime, timedelta
from random import randint
from time import sleep
from implementations.traders import SafeTrader

import matplotlib.pyplot as plt


class MatPlotLibVisualizer(Visualizer):

    def __init__(self, number_of_plotted_stocks=3, clear_interval=100):
        plt.ion()

        self.clear_interval = clear_interval * 2
        self.cycle = 0

        self.fig = plt.figure(constrained_layout=True)
        self.gs = self.fig.add_gridspec(2, number_of_plotted_stocks)

        self.plots = {}
        self.last_trading_data = {}
        self.color_mapping = {'orange': None, 'royalblue': None, 'peru': None, 'orchid': None}

        plt.subplots_adjust(left=0.1, bottom=None, right=0.9, top=None, wspace=0.2, hspace=None)

    def redraw(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.cycle += 1

        if self.cycle % self.clear_interval == 0:
            for plot in self.plots.values():
                plot.cla()

    def update_predictions_plot(self, data: List[Prediction]):
        stocks = {}

        for prediction in data:
            if prediction.stock in stocks:
                stocks[prediction.stock].append(prediction)
            else:
                stocks[prediction.stock] = [prediction]

        for index, stock in enumerate(stocks):
            if stock not in self.plots:
                self.plots[stock] = self.fig.add_subplot(self.gs[0, len(self.plots)])
                self.plots[stock].set_title(stock + " prediction")

            last_point = stocks[stock][len(stocks[stock]) - 1]
            self.plots[stock].set_xlim(last_point.current_timestamp - timedelta(minutes=15),
                                       last_point.predicted_timestamp)

            self.plots[stock].plot([p.current_timestamp for p in stocks[stock]],
                                   [p.current_price for p in stocks[stock]], 'b-')

            self.plots[stock].plot([p.predicted_timestamp for p in stocks[stock]],
                                   [p.predicted_price for p in stocks[stock]], 'r*-')

        self.redraw()

    def update_traders_plot(self, timestamp: datetime, traders: List[Trader], current_prices: Dict[str, float]):
        if 'trading' not in self.plots:
            self.plots['trading'] = self.fig.add_subplot(self.gs[1, :])
            self.plots['trading'].set_title("Traders' net worth")

        trading_data = {}
        for trader in traders:
            net_worth = trader.get_net_worth(timestamp, current_prices)
            trading_data[trader.name] = [timestamp, net_worth]

            color = next((c for c in self.color_mapping if self.color_mapping[c] == trader.name), None)
            if color is None:
                color = next((c for c in self.color_mapping if self.color_mapping[c] is None), None)
                self.color_mapping[color] = trader.name

            if trader.name not in self.last_trading_data:
                continue

            self.plots['trading'].plot([self.last_trading_data[trader.name][0], timestamp],
                                       [self.last_trading_data[trader.name][1], net_worth],
                                       label=trader.name, color=color)

        if self.plots['trading'].legend_ is None and self.last_trading_data != {}:
            self.plots['trading'].legend(loc="upper right")

        self.redraw()
        self.last_trading_data = trading_data


def test():
    # If this file is executed directly, the below examples will be run and tested:
    visualizer = MatPlotLibVisualizer()
    last_run = now = datetime.now()
    minute = timedelta(minutes=1)

    test_predictions = [
        Prediction('AAPL', now, now + minute, 100, 101),
        Prediction('TSLA', now, now + minute, 300, 296),
        Prediction('GME', now, now + minute, 30, 31),

        Prediction('AAPL', now + minute, now + minute * 2, 103, 105),
        Prediction('TSLA', now + minute, now + minute * 2, 290, 296),
        Prediction('GME', now + minute, now + minute * 2, 33, 35),
    ]

    last_predictions = test_predictions
    minutes_passed = 5

    while True:
        new_predictions = [
            Prediction('AAPL', now + minute * minutes_passed, now + minute * (minutes_passed + 1),
                       100 + randint(0, 10), 100 + randint(0, 10)),
            Prediction('TSLA', now + minute * minutes_passed, now + minute * (minutes_passed + 1),
                       280 + randint(0, 20), 280 + randint(0, 20)),
            Prediction('GME', now + minute * minutes_passed, now + minute * (minutes_passed + 1),
                       30 + randint(0, 5), 30 + randint(0, 5))
        ]

        visualizer.update_predictions_plot(last_predictions + new_predictions)
        visualizer.update_traders_plot(now + minute * minutes_passed, [SafeTrader("Trader 1", randint(500, 1000)),
                                                                       SafeTrader("Trader 2", randint(500, 1000))], {})

        last_predictions = new_predictions
        minutes_passed += 1
        sleep(1)

        print("Last run difference: ", datetime.now() - last_run)
        last_run = datetime.now()


if __name__ == '__main__':
    test()

