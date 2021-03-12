import findspark
from pyspark.sql import SparkSession

findspark.init()


def main():
    spark = SparkSession.builder.appName("Test").getOrCreate()
    sc = spark.sparkContext
    print("Running Spark version: ", sc.version)


if __name__ == '__main__':
    main()
