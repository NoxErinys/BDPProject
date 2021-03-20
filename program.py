from models import *
from implementations.data_collectors import *
from implementations.predictors import *
from implementations.traders import *
from implementations.visualizer import *
from time import sleep
from datetime import timedelta

TRADER_BALANCE = 1000
INTERVAL_IN_SECONDS = 60
NUMBER_OF_PREDICTION_PLOTS = 3
NUMBER_OF_TRAINING_DAYS = 10


def main(start_date):
    production_mode = (datetime.now() - start_date).days <= 1
    data_collector = YahooDataCollector(INTERVAL_IN_SECONDS)
    visualizer = MatPlotLibVisualizer()

    bots = [
        Bot(SafeTrader("Good LSTM Trader", TRADER_BALANCE), {}),
        Bot(SafeTrader("Bad LSTM Trader", TRADER_BALANCE), {})
    ]

    top_stocks = data_collector.get_top_stocks(start_date)

    for stock in top_stocks:
        stock_data = data_collector.get_historical_data(stock, start_date, NUMBER_OF_TRAINING_DAYS)

        bots[0].predictors[stock] = RNNPredictor(stock)
        bots[1].predictors[stock] = ImpairedRNNPredictor(stock)

        for bot in bots:
            bot.predictors[stock].train(stock_data)

    current_date = start_date
    end_date = start_date + timedelta(hours=8)
    last_predictions = []

    while current_date < end_date:
        latest_data = data_collector.get_latest_data_point(top_stocks, current_date)
        predictions_to_plot = []

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

        visualizer.update_predictions_plot(last_predictions + predictions_to_plot)
        visualizer.update_traders_plot(current_date, [b.trader for b in bots],
                                       {d[0]: d[1].close_price for d in latest_data.items()})

        last_predictions = predictions_to_plot

        if not production_mode:
            current_date += timedelta(seconds=INTERVAL_IN_SECONDS)
        else:
            execution_time = (datetime.now() - current_date).total_seconds()

            if INTERVAL_IN_SECONDS - execution_time > 0:
                sleep(INTERVAL_IN_SECONDS - execution_time + 1)

            current_date = datetime.now()


if __name__ == '__main__':
    main(datetime(2020, 5, 17))
