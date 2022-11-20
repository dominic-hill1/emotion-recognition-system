import pandas as pd

df = pd.read_csv("records.csv")
print(df["datetime"])