import pandas as pd

data = pd.read_csv("car_data.csv")

# print(data.head())


columns_to_drop = ["Name", "Power","Seats"]
data.drop(columns=columns_to_drop, inplace=True)

data = pd.get_dummies(data, columns=["Brand", "Fuel_Type", "Transmission", "Owner_Type"])
# data = data.astype(int)

print(data.columns.tolist())