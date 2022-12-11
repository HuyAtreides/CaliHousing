import pandas as pd

housing = pd.read_csv("housing.csv")

print(housing["ocean_proximity"].unique())