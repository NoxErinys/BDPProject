from typing import List, Dict
from datetime import datetime
from models import DataCollector, DataPoint


class YahooDataCollector(DataCollector):
    def __init__(self, start_date: datetime, interval_in_seconds: int):
        super().__init__(start_date, interval_in_seconds)

    def get_top_stocks(self, number_of_stocks=100) -> List[str]:
        return ["AAPL", "TSLA", "AMD"]
        # SAMI

    def get_historical_data(self, stock: str) -> List[DataPoint]:
        pass
        # SAMI

    def get_latest_data_point(self, stocks: List[str]) -> Dict[str, DataPoint]:
        pass
        # SAMI
