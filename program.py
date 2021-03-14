from models import *
from implementations.data_collectors import *
from implementations.predictors import *
from implementations.traders import *
from implementations.visualizer import *

from datetime import timedelta

TRADER_BALANCE = 1000
INTERVAL_IN_SECONDS = 60
NUMBER_OF_PREDICTION_PLOTS = 3


def main(start_date):
    data_collector = YahooDataCollector(start_date, INTERVAL_IN_SECONDS)
    visualizer = MatPlotLibVisualizer()

    bots = [
        Bot(SafeTrader("Good LSTM Trader", TRADER_BALANCE), {}),
        Bot(SafeTrader("Bad LSTM Trader", TRADER_BALANCE), {})
    ]

    predictions_to_plot = []
    top_stocks = data_collector.get_top_stocks()

    for stock in top_stocks:
        stock_data = data_collector.get_historical_data(stock)

        bots[0].predictors[stock] = RNNPredictor(stock)
        bots[1].predictors[stock] = ImpairedRNNPredictor(stock)

        for bot in bots:
            bot.predictors[stock].train(stock_data)

    current_date = start_date
    end_date = start_date + timedelta(hours=8)

    while current_date < end_date:
        latest_data = data_collector.get_latest_data_point(top_stocks)

        for bot in bots:
            predictions = []

            for index, stock in enumerate(top_stocks):
                current_data_point = latest_data[stock]
                predictor = bot.predictors[stock]

                predictor.update_model(current_data_point)
                prediction = predictor.predict_next_price(current_data_point)

                if index < NUMBER_OF_PREDICTION_PLOTS:
                    predictions_to_plot.append(prediction)

                predictions.append(prediction)

            bot.trader.trade(predictions)

        visualizer.plot_predictions(predictions_to_plot)
        visualizer.update_traders_plot(current_date, [b.trader for b in bots],
                                       {d[0]: d[1].close_price for d in latest_data.items()})

        current_date += timedelta(seconds=INTERVAL_IN_SECONDS)


if __name__ == '__main__':
    main(datetime(2020, 5, 17))
