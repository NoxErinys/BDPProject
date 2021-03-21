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

        if current_time < datetime.now() + timedelta(days=-1):
             return ['DECZ', 'ALACU', 'CRSAU', 'HMG', 'DBDRU', 'KTN', 'FEBZ', 'GSID', 'GECCM', 'RDIB', 'CMCTP', 'OCFCP', 'JBK', 'MACUU', 'GJT', 'LATNU', 'LSXMB', 'MAYS', 'LIVKU', 'ANDAU', 'IOR', 'CHPMU', 'GJH', 'GJR', 'ASRVP', 'RCHGU', 'CCZ', 'LACQU', 'SEPZ', 'DJUL', 'LEVLP', 'CGROU', 'ZGYHU', 'KTH', 'GGO', 'PSFD', 'FCRW', 'AIW', 'AMHCU', 'BROG', 'GFNCP', 'NAPR', 'PXSAP', 'TANNL', 'BRPAU', 'THCBU', 'BCYPU', 'GYRO', 'OBAS', 'TECTP', 'INBKL', 'GRNVU', 'PSCX', 'GLADL', 'REC', 'GSEE', 'THCAU', 'CFCV', 'ICCH', 'BLDG', 'TDACU', 'GJS', 'IBHF', 'LYFE', 'AVDG', 'HFBL', 'VBFC', 'IVLC', 'MDRRP', 'DGICB', 'PSMD', 'LUXE', 'MSVB', 'DIT', 'RILYI', 'MIG', 'EGIS', 'QADB', 'BWACU', 'PRTC', 'ELLO', 'OXSQZ', 'AVDR', 'MNSBP', 'JMPNL', 'IVSG', 'SLN', 'CKX', 'CBMB', 'CPHC', 'WVVIP', 'ARGD', 'ZIONN', 'BDL', 'MRSK', 'SVT', 'SAF', 'IVDG', 'WHLRP', 'SAK'][:number_of_stocks]

        # Spark dataframe will be populated with all stocks data
        spark_data_frame = self.spark.createDataFrame([], self.schema)
        pandas_data_frame = pandas\
            .DataFrame(columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'AdjustedClose', 'Volume', 'Symbol'])

        lookup_table = pandas.read_csv("../datasets/nasdaq_lookup_table/nasdaq_lookup.csv", sep=";").sort_values("Volume", ascending=False).head(500)
        stocks_list = [stock for stock in lookup_table["Symbol"].tolist() if "^" not in stock and "/" not in stock]
        download_list = []

        number_of_days_to_analyse = 7
        for i in range(0, number_of_days_to_analyse):
            start_time = (current_time - timedelta(days=i + 1)).strftime("%Y-%m-%d")
            end_time = (current_time - timedelta(days=i + 1)).strftime("%Y-%m-%d")

            historical_data_path = "../datasets/historical_data_day/"
            folder_path = historical_data_path + start_time + "/"
            try:
                Path(folder_path).mkdir(parents=True, exist_ok=True)
            except OSError:
                print(f'Creation of the directory {folder_path} failed')

            for stock in stocks_list:
                stock_file = Path(folder_path + stock + ".csv")
                if stock_file.is_file():
                    # file exists
                    # stock_data_spark_df = self.spark.read .csv(str(stock_file), schema=self.schema, timestampFormat="yyyy-MM-dd HH:mm:ss", header=True)
                    # spark_data_frame = spark_data_frame.union(stock_data_spark_df)
                    stock_data = pandas.read_csv(str(stock_file))
                    stock_data['Datetime'] = pandas.to_datetime(stock_data['Datetime'], format='%Y-%m-%d')

                    pandas_data_frame = pandas.concat([pandas_data_frame, stock_data])
                else:
                    # add to download list
                    download_list.append(stock)

            if len(download_list) > 0:
                stocks_data = yf.download(download_list, start=start_time, end=end_time, interval="1d", group_by="ticker")
                if len(stocks_data) > 0:
                    for stock in download_list:
                        stock_data = stocks_data if len(download_list) == 1 else stocks_data[stock]
                        stock_data = stock_data.rename(columns={"Adj Close": "AdjustedClose"})
                        stock_data = stock_data.reset_index()
                        stock_data.dropna(inplace=True)
                        stock_data = stock_data.rename(columns={"Date": "Datetime"})
                        stock_data["Volume"] = stock_data["Volume"].astype(float)
                        stock_data["Symbol"] = stock
                        stock_data.set_index('Datetime')
                        stock_data['Datetime'] = pandas.to_datetime(stock_data['Datetime'], format='%Y-%m-%d')
                        stock_file = Path(folder_path + stock + ".csv")
                        # if stock_data.size > 0:
                        stock_data.to_csv(path_or_buf=stock_file, index=False)
                        pandas_data_frame = pandas.concat([pandas_data_frame, stock_data])

                        # stock_data_spark_df = self.spark.createDataFrame(stock_data, self.schema)
                        # spark_data_frame = spark_data_frame.union(stock_data_spark_df)

                else:
                    print(f'stocks data not found on yahoo finance')
                    continue

        # print(pandas_data_frame)
        # try to get top stocks
        top_stocks_list = self.spark.createDataFrame(pandas_data_frame, self.schema).groupBy("Symbol") \
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

            historical_data_path = "../datasets/historical_data/"
            folder_path = historical_data_path + start_time + "/"
            try:
                Path(folder_path).mkdir(parents=True, exist_ok=True)
            except OSError:
                print(f'Creation of the directory {folder_path} failed')
            # else:
                # print(f'Successfully created the directory {folder_path}')

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
                if current_time - timedelta(days=i) < datetime.now() + timedelta(days=-1):
                    stock_data.to_csv(path_or_buf=stock_file, index=False)
                stock_data_spark_df = self.spark.createDataFrame(stock_data, self.schema)
                spark_data_frame_for_stock = spark_data_frame_for_stock.union(stock_data_spark_df)

        spark_data_frame_for_stock_sorted = spark_data_frame_for_stock\
            .where(spark_data_frame_for_stock.Datetime <= current_time.strftime("%Y-%m-%d %H:%M:%S"))\
            .sort("Datetime")\
            .collect()

        list_of_data_points = [DataPoint(row.Open,
                                         row.Close,
                                         row.High,
                                         row.Low,
                                         row.Volume,
                                         row.Datetime)
                               for row in spark_data_frame_for_stock_sorted]

        return list_of_data_points

    def get_latest_data_point(self, stocks: List[str], current_time: datetime) -> Dict[str, DataPoint]:

        spark_data_frame_for_stock = self.spark.createDataFrame([], self.schema)
        pandas_data_frame = pandas\
            .DataFrame(columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'AdjustedClose', 'Volume', 'Symbol'])

        download_list = []

        stocks_dict = {}

        start_time = current_time.strftime("%Y-%m-%d")
        end_time = (current_time + timedelta(days=1)).strftime("%Y-%m-%d")

        historical_data_path = "../datasets/historical_data/"
        folder_path = historical_data_path + start_time + "/"

        for stock in stocks:
            try:
                Path(folder_path).mkdir(parents=True, exist_ok=True)
            except OSError:
                print(f'Creation of the directory {folder_path} failed')

            stock_file = Path(folder_path + stock + ".csv")
            if stock_file.is_file() and current_time < datetime.now() + timedelta(hours=-24):
                # if stock data already downloaded, just load it
                # stock_data_spark_df = self.spark.read .csv(str(stock_file), schema=self.schema, timestampFormat="yyyy-MM-dd HH:mm:ss", header=True)
                # spark_data_frame_for_stock = spark_data_frame_for_stock.union(stock_data_spark_df)
                stock_data = pandas.read_csv(str(stock_file))
                stock_data['Datetime'] = pandas.to_datetime(stock_data['Datetime'], format='%Y-%m-%d')
                pandas_data_frame = pandas.concat([pandas_data_frame, stock_data])

            else:
                # add stock to download list
                download_list.append(stock)

        if len(download_list) > 0:
            stocks_data = yf.download(download_list, start=start_time, end=end_time, interval="1m")
            if len(stocks_data) > 0:
                for stock in download_list:

                    stock_data = stocks_data if len(download_list) == 1 else stocks_data[stock]
                    stock_data = stock_data.rename(columns={"Adj Close": "AdjustedClose"})
                    stock_data = stock_data.reset_index()
                    stock_data.dropna(inplace=True)
                    stock_data["Datetime"] = stock_data["Datetime"].astype(str).str[:-6].astype('datetime64[ns]')
                    stock_data["Volume"] = stock_data["Volume"].astype(float)
                    stock_data["Symbol"] = stock
                    stock_data.set_index('Datetime')

                    if current_time < datetime.now() + timedelta(hours=-24):
                        stock_file = Path(folder_path + stock + ".csv")
                        stock_data.to_csv(path_or_buf=stock_file, index=False)

                    pandas_data_frame = pandas.concat([pandas_data_frame, stock_data])

            # date_filter_string = "Datetime < '" + current_time.strftime("%Y-%m-%d %H:%M:%S") + "'"
            # spark_data_frame_for_stock = spark_data_frame_for_stock.union(self.spark.createDataFrame(stock_data, self.schema))
            # data_frame = self.spark.createDataFrame(stock_data, self.schema)

        print(pandas_data_frame)
        spark_data_frame_for_stock = self.spark.createDataFrame(pandas_data_frame, self.schema)
        for stock in stocks:
            last_point_row = spark_data_frame_for_stock \
                .where(spark_data_frame_for_stock.Datetime <= current_time.strftime("%Y-%m-%d %H:%M:%S")) \
                .where(spark_data_frame_for_stock.Symbol == stock)\
                .sort("Datetime", ascending=False) \
                .limit(1) \
                .select("*") \
                .first()

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

    # print(yahoo_data_collector.get_top_stocks(datetime(2021, 3, 21), 4))

    # list_of_data_points = yahoo_data_collector.get_historical_data("AAPL", datetime(2021, 3, 17), 2)
    # print([f'{data_point.timestamp}| volume: {data_point.volume}, close: {data_point.close_price}' for data_point in list_of_data_points])

    data_point = yahoo_data_collector.get_latest_data_point(["AAPL", "AMZN"], datetime(2021, 3, 19, 14, 30, 21))
    print(f'AAPL: {data_point["AAPL"].timestamp}| volume: {data_point["AAPL"].volume}, close: {data_point["AAPL"].close_price}')
    print(f'AMZN: {data_point["AMZN"].timestamp}| volume: {data_point["AMZN"].volume}, close: {data_point["AMZN"].close_price}')


if __name__ == '__main__':
    test()
