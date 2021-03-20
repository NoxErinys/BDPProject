import yfinance as yf
import pandas
import findspark
from functools import reduce
from typing import List, Dict
from datetime import datetime, timedelta
from models import DataCollector, DataPoint
from os import mkdir
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, IntegerType


class YahooDataCollector(DataCollector):
    def __init__(self, interval_in_seconds: int):
        findspark.init()
        self.spark = SparkSession.builder.appName("BDPProject").getOrCreate()
        self.sc = self.spark.sparkContext
        self.schema = StructType([
            StructField('Datetime', TimestampType(), True),
            StructField('Open', DoubleType(), True),
            StructField('High', DoubleType(), True),
            StructField('Low', DoubleType(), True),
            StructField('Close', DoubleType(), True),
            StructField('AdjustedClose', DoubleType(), True),
            StructField('Volume', DoubleType(), True),
            StructField('Symbol', StringType(), True),
        ])

        super().__init__(interval_in_seconds)

    def get_top_stocks(self, current_time: datetime, number_of_stocks=100) -> List[str]:

        # Spark dataframe will be populated with all stocks data
        spark_data_frame = self.spark.createDataFrame([], self.schema)

        lookup_table = pandas.read_csv("./datasets/nasdaq_lookup_table/nasdaq_lookup.csv", sep=";")
        stocks_list = lookup_table["Symbol"].tolist()

        number_of_days_to_analyse = 30
        for i in range(number_of_days_to_analyse):
            start_time = (current_time - timedelta(days=i + 1)).strftime("%Y-%m-%d")
            end_time = (current_time - timedelta(days=i)).strftime("%Y-%m-%d")

            historical_data_path = "./datasets/historical_data/"
            folder_path = historical_data_path + start_time + "/"
            try:
                mkdir(folder_path)
            except OSError:
                print(f'Creation of the directory {folder_path} failed')
            else:
                print(f'Successfully created the directory {folder_path}')

            for stock in stocks_list:
                if "^" in stock or "/" in stock:
                    # stock name contains ^ or /
                    print(f'stock name contains ^ or /: {stock}')
                    continue

                stock_file = Path(folder_path + stock + ".csv")
                if stock_file.is_file():
                    # file exists
                    stock_data_spark_df = self.spark.read \
                        .csv(str(stock_file), schema=self.schema, timestampFormat="yyyy-MM-dd HH:mm:ss", header=True)

                    spark_data_frame = spark_data_frame.union(stock_data_spark_df)

                    # print(f'file exists: {stock}.csv, data loaded to spark!')

                else:
                    stock_data = yf.download(stock, start=start_time, end=end_time, interval="1m")
                    if len(stock_data) < 1:
                        print(f'stock data not found on yahoo finance: {stock}')
                        continue

                    stock_data = stock_data.rename(columns={"Adj Close": "AdjustedClose"})
                    stock_data = stock_data.reset_index()
                    stock_data.dropna(inplace=True)
                    stock_data["Datetime"] = stock_data["Datetime"].astype(str).str[:-6].astype('datetime64[ns]')
                    stock_data["Volume"] = stock_data["Volume"].astype(float)

                    # stock_data.insert('Symbol', stock)
                    stock_data["Symbol"] = stock

                    stock_data.set_index('Datetime')
                    stock_data.to_csv(path_or_buf=stock_file, index=False)

                    stock_data_spark_df = self.spark.createDataFrame(stock_data, self.schema)
                    spark_data_frame = spark_data_frame.union(stock_data_spark_df)

        # try to get top stocks
        top_stocks_list = spark_data_frame.groupBy("Symbol") \
            .agg({'Volume': 'avg'}) \
            .sort("avg(Volume)") \
            .select("Symbol") \
            .limit(number_of_stocks) \
            .rdd.flatMap(lambda x: x).collect()
        return top_stocks_list

    def get_historical_data(self, stock: str, current_time: datetime, number_of_days: int = 10) -> List[DataPoint]:
        spark_data_frame_for_stock = self.spark.createDataFrame([], self.schema)

        for i in range(number_of_days + 1):
            start_time = (current_time - timedelta(days=i)).strftime("%Y-%m-%d")
            end_time = (current_time - timedelta(days=i - 1)).strftime("%Y-%m-%d")

            historical_data_path = "./datasets/historical_data/"
            folder_path = historical_data_path + start_time + "/"
            try:
                mkdir(folder_path)
            except OSError:
                print(f'Creation of the directory {folder_path} failed')
            else:
                print(f'Successfully created the directory {folder_path}')

            stock_file = Path(folder_path + stock + ".csv")
            if stock_file.is_file():
                # if stock data already downloaded, just load it
                stock_data_spark_df = self.spark.read \
                    .csv(str(stock_file), schema=self.schema, timestampFormat="yyyy-MM-dd HH:mm:ss", header=True)
                spark_data_frame_for_stock = spark_data_frame_for_stock.union(stock_data_spark_df)
            else:
                # download if not downloaded
                stock_data = yf.download(stock, start=start_time, end=end_time, interval="1m")
                if len(stock_data) < 1:
                    print(f'stock data not found on yahoo finance: {stock}')
                    continue

                stock_data = stock_data.rename(columns={"Adj Close": "AdjustedClose"})
                stock_data = stock_data.reset_index()
                stock_data.dropna(inplace=True)
                stock_data["Datetime"] = stock_data["Datetime"].astype(str).str[:-6].astype('datetime64[ns]')
                stock_data["Volume"] = stock_data["Volume"].astype(float)
                stock_data["Symbol"] = stock
                stock_data.set_index('Datetime')
                stock_data.to_csv(path_or_buf=stock_file, index=False)
                stock_data_spark_df = self.spark.createDataFrame(stock_data, self.schema)
                spark_data_frame_for_stock = spark_data_frame_for_stock.union(stock_data_spark_df)

        spark_data_frame_for_stock_sorted = spark_data_frame_for_stock.sort("Datetime").collect()
        list_of_data_points = [DataPoint(row.Open,
                                         row.Close,
                                         row.High,
                                         row.Low,
                                         row.Volume,
                                         row.Datetime)
                               for row in spark_data_frame_for_stock_sorted]

        return list_of_data_points

    def get_latest_data_point(self, stocks: List[str], current_time: datetime) -> Dict[str, DataPoint]:

        stocks_dict = {}

        start_time = current_time.strftime("%Y-%m-%d")
        end_time = (current_time + timedelta(days=1)).strftime("%Y-%m-%d")

        for stock in stocks:
            stock_data = yf.download(stock, start=start_time, end=end_time, interval="1m")
            if len(stock_data) < 1:
                print(f'stock data not found on yahoo finance: {stock}')
                continue

            stock_data = stock_data.rename(columns={"Adj Close": "AdjustedClose"})
            stock_data = stock_data.reset_index()
            stock_data.dropna(inplace=True)
            stock_data["Datetime"] = stock_data["Datetime"].astype(str).str[:-6].astype('datetime64[ns]')
            stock_data["Volume"] = stock_data["Volume"].astype(float)
            stock_data["Symbol"] = stock
            stock_data.set_index('Datetime')

            last_point_row = self.spark.createDataFrame(stock_data, self.schema) \
                .sort("Datetime", ascending=False).limit(1).select("*").first()
            data_point = DataPoint(last_point_row.Open,
                                   last_point_row.Close,
                                   last_point_row.High,
                                   last_point_row.Low,
                                   last_point_row.Volume,
                                   last_point_row.Datetime)
            stocks_dict[stock] = data_point

        return stocks_dict


def test():
    yahoo_data_collector = YahooDataCollector(60)
    list_of_data_points = yahoo_data_collector.get_historical_data("AAPL", datetime(2021, 3, 17), 2)
    print([f'{data_point.timestamp}| volume: {data_point.volume}, close: {data_point.close_price}' for data_point in list_of_data_points])


if __name__ == '__main__':
    test()
