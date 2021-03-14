import findspark
from pyspark.sql import SparkSession

findspark.init()


def main():
    spark = SparkSession.builder.appName("Test").getOrCreate()
    sc = spark.sparkContext
    print("Running Spark version: ", sc.version)


def plot():
    from datetime import datetime, timedelta
    from matplotlib import pyplot
    from matplotlib.animation import FuncAnimation
    from random import randrange

    x_data, y_data = [], []
    width = timedelta(seconds=10)

    figure = pyplot.figure()
    ax = pyplot.axes(xlim=(0, 100), ylim=(0, 100))
    line, = pyplot.plot_date(x_data, y_data, '-')

    def update(frame):
        time_now = datetime.now()
        x_data.append(datetime.now())
        y_data.append(randrange(0, 100))
        line.set_data(x_data, y_data)

        ax.set_xlim(time_now - width, datetime.now())  # added ax attribute here
        figure.gca().relim()
        figure.gca().autoscale_view()
        return line,

    animation = FuncAnimation(figure, update, interval=10)

    pyplot.show()


if __name__ == '__main__':
    main()
