import pandas as pd

path = "data/btc_30m.csv"

data = pd.read_csv(path, parse_dates=["Time"])
data = data[["Time", "Open", "High", "Low", "Close", "Volume"]]

data.to_csv(path, index=False)