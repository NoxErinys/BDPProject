from models import *
import seqlog
import logging
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
    visualizer = None if production_mode else \
        MatPlotLibVisualizer(NUMBER_OF_PREDICTION_PLOTS, show_interval=timedelta(minutes=20))

    bots = [
        Bot(SafeTrader("Good LSTM Trader", TRADER_BALANCE), {}),
        Bot(SafeTrader("Bad LSTM Trader", TRADER_BALANCE), {})
    ]

    top_stocks = data_collector.get_top_stocks(start_date, number_of_stocks=100)

    for stock in top_stocks:
        stock_data = data_collector.get_historical_data(stock, start_date, NUMBER_OF_TRAINING_DAYS)

        bots[0].predictors[stock] = RNNPredictor(stock, timedelta(seconds=INTERVAL_IN_SECONDS))
        bots[1].predictors[stock] = ImpairedRNNPredictor(stock, timedelta(seconds=INTERVAL_IN_SECONDS))

        for bot in bots:
            bot.predictors[stock].train(stock_data)

    current_date = start_date
    end_date = start_date + timedelta(hours=8)
    last_predictions = []

    while current_date < end_date:
        latest_data = data_collector.get_latest_data_point(top_stocks, current_date)
        stock_prices = {d[0]: d[1].close_price for d in latest_data.items()}
        predictions_to_plot = []

        for bot_index, bot in enumerate(bots):
            predictions = []

            for index, stock in enumerate(top_stocks):
                current_data_point = latest_data[stock]
                predictor = bot.predictors[stock]

                if current_date > start_date:
                    predictor.update_model(current_data_point)

                prediction = predictor.predict_next_price(current_data_point)

                if not production_mode and index < NUMBER_OF_PREDICTION_PLOTS and bot_index == 0:
                    predictions_to_plot.append(prediction)

                predictions.append(prediction)

            bot.trader.trade(predictions)
            logging.info("{trader_name} has now a net worth of {net_worth}",
                         bot.trader.name, bot.trader.get_current_net_worth(stock_prices))

        if not production_mode:
            visualizer.update_predictions_plot(last_predictions + predictions_to_plot)
            visualizer.update_traders_plot(current_date, [b.trader for b in bots], stock_prices)

            last_predictions = predictions_to_plot

            current_date += timedelta(seconds=INTERVAL_IN_SECONDS)
        else:
            execution_time = (datetime.now() - current_date).total_seconds()

            if INTERVAL_IN_SECONDS - execution_time > 0:
                sleep(INTERVAL_IN_SECONDS - execution_time + 1)

            current_date = datetime.now()


if __name__ == '__main__':
    seqlog.log_to_seq(
        server_url="http://localhost:5341/",
        api_key="",
        level=logging.INFO,
        batch_size=10,
        auto_flush_timeout=10,
        override_root_logger=True,
    )

    main(datetime(2021, 3, 18))
