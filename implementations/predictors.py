from typing import List
from datetime import datetime, timedelta
from models import Prediction, DataPoint, Predictor
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class RNNPredictor(Predictor):

    def __init__(self, stock: str, window: int, epochs: int):
        super().__init__(stock, window, epochs)

        self.last_batch = []

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dense(units=1)
        ])

        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')
        self.model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(),
                           metrics=[tf.metrics.MeanAbsoluteError()])

    def train(self, data: List[DataPoint]):
        ordered_data = sorted(data, key=lambda x: x.timestamp)
        ordered_data = [self.data_point_to_array(d) for d in ordered_data]
        print("Number of inputs: ", len(ordered_data))

        n = len(ordered_data)
        train_data = ordered_data[0:int(n * 0.7)]
        val_data = ordered_data[int(n * 0.7):int(n * 0.9)]
        test_data = ordered_data[int(n * 0.9):]

        self.model.fit(self.make_dataset(train_data), validation_data=self.make_dataset(val_data),
                       epochs=self.epochs, callbacks=[self.early_stopping])

        self.last_batch = ordered_data[-1 * self.window:]

        inputs, labels = next(iter(self.make_dataset(test_data)))

        print(len(inputs))
        print(len(labels))

        plt.plot([i for i in range(self.window - 1)],
                 labels[0, :, 0], 'b-', label="Actual")

        predictions = self.model(inputs)
        plt.plot([i for i in range(self.window - 1)],
                 predictions[0, :, 0], 'r-*', label="Predictions")
        plt.legend()
        plt.show()

    def predict_next_price(self, current_data_point: DataPoint) -> Prediction:
        pass
        # BOTH, INDIVIDUALLY

    def update_model(self, actual_data_point: DataPoint):
        pass
        # BOTH, INDIVIDUALLY

    def data_point_to_array(self, point: DataPoint):
        return [point.timestamp.minute, point.volume, point.close_price]

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.window,
            sequence_stride=1,
            shuffle=False,
            batch_size=32, )

        ds = ds.map(self.split_window)
        return ds

    def split_window(self, features):
        inputs = features[:, slice(0, self.window - 1), :]
        labels = features[:, slice(1, None), :]
        labels = tf.stack([labels[:, :, 2]], axis=-1)  # index 2 = close_price

        inputs.set_shape([None, self.window - 1, None])
        labels.set_shape([None, self.window - 1, None])

        return inputs, labels


def test():
    from implementations.data_collectors import YahooDataCollector

    data_collector = YahooDataCollector(60)
    historical_data = data_collector.get_historical_data('AAPL', datetime(2021, 3, 5, 10, 15))

    """
    time_points = [
        data_collector.get_latest_data_point(['AAPL'], datetime(2021, 3, 5, 10, 16))['AAPL'],
        data_collector.get_latest_data_point(['AAPL'], datetime(2021, 3, 5, 10, 17))['AAPL'],
        data_collector.get_latest_data_point(['AAPL'], datetime(2021, 3, 5, 10, 18))['AAPL'],
        data_collector.get_latest_data_point(['AAPL'], datetime(2021, 3, 5, 10, 19))['AAPL'],
        data_collector.get_latest_data_point(['AAPL'], datetime(2021, 3, 5, 10, 20))['AAPL'],
        data_collector.get_latest_data_point(['AAPL'], datetime(2021, 3, 5, 10, 21))['AAPL'],
        data_collector.get_latest_data_point(['AAPL'], datetime(2021, 3, 5, 10, 22))['AAPL'],
        data_collector.get_latest_data_point(['AAPL'], datetime(2021, 3, 5, 10, 23))['AAPL'],
        data_collector.get_latest_data_point(['AAPL'], datetime(2021, 3, 5, 10, 24))['AAPL'],
        data_collector.get_latest_data_point(['AAPL'], datetime(2021, 3, 5, 10, 25))['AAPL']
    ]
    """

    predictor = RNNPredictor('AAPL', 25, 10)
    predictor.train(historical_data)


if __name__ == '__main__':
    test()
