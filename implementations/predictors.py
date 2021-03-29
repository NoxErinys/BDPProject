from typing import List
from datetime import datetime, timedelta
from models import Prediction, DataPoint, Predictor
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

LABEL_INDEX = 2


class RNNPredictor(Predictor):

    def __init__(self, stock: str, data_interval: timedelta, window: int=100, epochs: int=40, dev_mode: bool=False):
        super().__init__(stock, data_interval, window, epochs)
        self.dev_mode = dev_mode

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dense(units=1)
        ])

        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
        self.early_stopping_single = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, mode='min')
        self.model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(),
                           metrics=[tf.metrics.MeanAbsoluteError()])

        self.mean = []
        self.std = []
        self.last_batch = []

    def train(self, data: List[DataPoint]):
        ordered_data = sorted(data, key=lambda x: x.timestamp)
        ordered_data = np.array([self.data_point_to_array(d) for d in ordered_data], dtype=np.float32)
        print("Number of inputs: ", len(ordered_data))

        if self.dev_mode:
            # Run some tests
            n = len(ordered_data)
            train_data = ordered_data[0:int(n * 0.7)]
            val_data = ordered_data[int(n * 0.7):int(n * 0.9)]
            test_data = ordered_data[int(n * 0.9):]

            self.mean = np.mean(train_data, axis=0)
            self.std = np.std(train_data, axis=0)

            train_data = (train_data - self.mean) / self.std
            val_data = (val_data - self.mean) / self.std
            test_data = (test_data - self.mean) / self.std

            self.model.fit(self.make_dataset(train_data), validation_data=self.make_dataset(val_data),
                           epochs=self.epochs, callbacks=[self.early_stopping])

            inputs, labels = next(iter(self.make_dataset(test_data)))
            plt.plot([i for i in range(self.window - 1)],
                     labels[0, :, 0] * self.std[2] + self.mean[2], 'b-', label="Actual")

            predictions = self.model(inputs)
            plt.plot([i for i in range(self.window - 1)],
                     predictions[0, :, 0] * self.std[2] + self.mean[2], 'r-*', label="Predictions")

            actual = [x[0][0] for x in labels.numpy()]
            actual_change = []
            predicted = [x[0][0] for x in predictions.numpy()]
            predicted_change = []

            for i in range(1, len(actual)):
                actual_change_value = (actual[i] - actual[i - 1])/abs(actual[i] - actual[i - 1])
                actual_change.append(actual_change_value)
                predicted_change_value = (predicted[i] - predicted[i - 1])/abs(predicted[i] - predicted[i - 1])
                predicted_change.append(predicted_change_value)

            matches = [1 for x, y in zip(actual_change, predicted_change) if x == y]
            print(f'input labels= {actual}')
            print(f'input labels change= {actual_change}')
            print(f'predictions= {predicted}')
            print(f'input labels change= {predicted_change}')
            print(f'matches={matches}')
            print(f'trend accuracy = {len(matches)/len(actual_change)}')
            plt.legend()
            plt.show()

        else:
            train_data = ordered_data[0:int(len(ordered_data) * 0.8)][:]
            val_data = ordered_data[int(len(ordered_data) * 0.8):][:]

            self.mean = np.mean(train_data, axis=0)
            self.std = np.std(train_data, axis=0)

            train_data = (train_data - self.mean) / self.std
            val_data = (val_data - self.mean) / self.std
            ordered_data = (ordered_data - self.mean) / self.std

            self.model.fit(self.make_dataset(train_data), validation_data=self.make_dataset(val_data),
                           epochs=self.epochs, callbacks=[self.early_stopping])

        self.last_batch = ordered_data[-1 * (self.window - 1):]

    def predict_next_price(self, current_data_point: DataPoint) -> Prediction:
        inputs, labels = next(iter(self.make_dataset(np.concatenate((self.last_batch,
                                                                    [self.data_point_to_array(current_data_point)])))))
        predictions = self.model(inputs).numpy()
        last_batch = predictions[len(predictions) - 1]
        prediction = last_batch[len(last_batch) - 1][0]

        return Prediction(self.stock, current_data_point.timestamp, current_data_point.timestamp + self.data_interval,
                          current_data_point.close_price, prediction * self.std[LABEL_INDEX] + self.mean[LABEL_INDEX])

    def update_model(self, actual_data_point: DataPoint):
        self.last_batch = np.concatenate((self.last_batch, [self.data_point_to_array(actual_data_point)]))

        # This should add on-top of the existing model, according to this:
        # https://github.com/keras-team/keras/issues/4446
        self.model.fit(self.make_dataset(self.last_batch), epochs=int(self.epochs / 2),
                       callbacks=[self.early_stopping_single])

        self.last_batch = self.last_batch[1:len(self.last_batch)]

    def data_point_to_array(self, point: DataPoint):
        point = np.array([point.timestamp.minute, point.volume, point.close_price, point.open_price,
                          point.lowest_price, point.highest_price])

        if len(self.std) == 0:
            return point
        else:
            return (point - self.mean) / self.std

    def make_dataset(self, data):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.window,
            sequence_stride=1,
            shuffle=False,
            batch_size=64, )

        ds = ds.map(self.split_window)
        return ds

    def split_window(self, features):
        inputs = features[:, slice(0, self.window - 1), :]
        labels = features[:, slice(1, None), :]
        labels = tf.stack([labels[:, :, LABEL_INDEX]], axis=-1)

        inputs.set_shape([None, self.window - 1, None])
        labels.set_shape([None, self.window - 1, None])

        return inputs, labels


class ImpairedRNNPredictor(RNNPredictor):

    def __init__(self, stock: str, data_interval: timedelta, window: int=100, epochs: int=32, dev_mode: bool=False):
        super().__init__(stock, data_interval, window, epochs, dev_mode)

    def update_model(self, actual_data_point: DataPoint):
        self.last_batch = np.concatenate((self.last_batch, [self.data_point_to_array(actual_data_point)]))
        self.last_batch = self.last_batch[1:len(self.last_batch)]


def test():
    from implementations.data_collectors import YahooDataCollector
    from implementations.visualizer import MatPlotLibVisualizer
    from time import sleep

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    number_of_predictions = 400
    stock = 'GME'
    date = datetime(2021, 3, 18, 9, 35)

    # visualizer = MatPlotLibVisualizer(1, show_interval=timedelta(minutes=number_of_predictions))
    data_collector = YahooDataCollector(60)
    historical_data = data_collector.get_historical_data(stock, date, number_of_days=25)

    predictor = ImpairedRNNPredictor(stock, timedelta(minutes=1), 100, 35, dev_mode=False)
    predictor.train(historical_data)

    predictions = []

    for index in range(number_of_predictions):
        point = data_collector.get_latest_data_point([stock], date + timedelta(minutes=index + 1))[stock]

        if index > 0:
            predictor.update_model(point)

        prediction = predictor.predict_next_price(point)
        predictions.append(prediction)
        # visualizer.update_predictions_plot(predictions)

    actual_change = []
    predicted_change = []
    error = []

    for i in range(1, len(predictions) - 1):
        actual_change_value = (predictions[i + 1].current_price - predictions[i].current_price) / abs(predictions[i + 1].current_price - predictions[i].current_price)
        actual_change.append(actual_change_value)
        predicted_change_value = (predictions[i].predicted_price - predictions[i - 1].predicted_price) / abs(predictions[i].predicted_price - predictions[i - 1].predicted_price)
        predicted_change.append(predicted_change_value)
        error.append(abs(predictions[i].predicted_price - predictions[i + 1].current_price))

    matches = [1 for x, y in zip(actual_change, predicted_change) if x == y]
    print(f'input labels change= {actual_change}')
    print(f'predicted change change= {predicted_change}')
    print(f'matches={matches}')
    print(f'trend accuracy = {len(matches) / len(actual_change)}')
    print(f'error = {sum(error) / len(error) / (sum([x.current_price for x in predictions]) / len(predictions))}')

    while True:
        sleep(1)


if __name__ == '__main__':
    test()
