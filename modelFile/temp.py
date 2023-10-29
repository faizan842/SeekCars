import pandas as pd

data = pd.read_csv("car_data.csv")

data.sort_values(by='New_Price', inplace=True)

data.to_csv('new_car_data.csv', index=False)