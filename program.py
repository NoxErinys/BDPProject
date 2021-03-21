from models import *
import seqlog
import logging
import tensorflow as tf
from implementations.data_collectors import *
from implementations.predictors import *
from implementations.traders import *
from implementations.visualizer import *
from time import sleep
from datetime import timedelta

TRADER_BALANCE = 10000
INTERVAL_IN_SECONDS = 60
NUMBER_OF_PREDICTION_PLOTS = 3
NUMBER_OF_TRAINING_DAYS = 25
NUMBER_OF_EPOCHS = 35
NUMBER_OF_STOCKS = 12


def copy_model(model_source, model_target):
    for l_tg, l_sr in zip(model_target.layers, model_source.layers):
        wk0 = l_sr.get_weights()
        l_tg.set_weights(wk0)


def main(start_date):
    logging.info("Starting traders bots...")

    production_mode = (datetime.now() - start_date).days <= 1
    data_collector = YahooDataCollector(INTERVAL_IN_SECONDS)
    visualizer = None if production_mode else \
        MatPlotLibVisualizer(NUMBER_OF_PREDICTION_PLOTS, show_interval=timedelta(minutes=20))

    bots = [
        Bot(SimpleTrader("Continuous RNN Trader", TRADER_BALANCE), {}),
        Bot(SimpleTrader("Impaired RNN Trader", TRADER_BALANCE), {})
    ]

    top_stocks = data_collector.get_top_stocks(start_date, number_of_stocks=NUMBER_OF_STOCKS)

    logging.info("Training trading bots...")

    for stock in top_stocks:
        stock_data = data_collector.get_historical_data(stock, start_date, NUMBER_OF_TRAINING_DAYS)

        bots[0].predictors[stock] = RNNPredictor(stock, timedelta(seconds=INTERVAL_IN_SECONDS), epochs=NUMBER_OF_EPOCHS)
        bots[1].predictors[stock] = ImpairedRNNPredictor(stock, timedelta(seconds=INTERVAL_IN_SECONDS), epochs=NUMBER_OF_EPOCHS)

        for bot in bots:
            bot.predictors[stock].train(stock_data)

    current_date = start_date
    end_date = start_date + timedelta(hours=8)
    last_predictions = []

    logging.info("Trading bots are now running...")
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
                         trader_name=bot.trader.name, net_worth=bot.trader.get_current_net_worth(stock_prices))

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

    logging.info("Trading bots have finished trading")
    for bot in bots:
        bot.trader.save_data(f"datasets/traders/{bot.trader.name}.json")

    logging.info("Successfully saved trading bots' data")


if __name__ == '__main__':
    seqlog.log_to_seq(
        server_url="http://localhost:5341/",
        api_key="",
        level=logging.INFO,
        batch_size=10,
        auto_flush_timeout=10,
        override_root_logger=True,
    )

    main(datetime(2021, 3, 18, 9, 35))
