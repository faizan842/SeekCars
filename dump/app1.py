import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import pickle

data = pd.read_csv("car_data.csv")

columns_to_drop = ["Name", "Power","Seats"]
data.drop(columns=columns_to_drop, inplace=True)

data = pd.get_dummies(data, columns=["Brand", "Fuel_Type", "Transmission", "Owner_Type"])


cityData = {
    "Kolkata": 1,
    "Ahmedabad": 2,
    "Delhi": 3,
    "Jaipur": 4,
    "Coimbatore": 5,
    "Chennai": 6,
    "Pune": 7,
    "Mumbai": 8,
    "Kochi": 9,
    "Hyderabad": 10,
    "Bangalore": 11
}

data['Location'] = data['Location'].replace(cityData)


current_year = 2023
data["Age"] = current_year - data["Year"]

# maxKm =  data['Kilometers_Driven'].max()
# print(maxKm)
# print(21000/maxKm)0.09767441860465116
# data['Kilometers_Driven'] = data['Kilometers_Driven'] / maxKm
data['Kilometers_Driven'] = data['Kilometers_Driven'] / 1000
data['Engine'] = data['Engine'] / 100


data = data.astype(int)
print(data.head())




X = data[['Kilometers_Driven', 'Mileage', 'Engine', 'New_Price', 'Location', 'Brand_Audi', 'Brand_BMW', 'Brand_Ford', 'Brand_Honda', 'Brand_Hyundai', 'Brand_Mahindra', 'Brand_Maruti', 'Brand_Mercedes', 'Brand_Nissan', 'Brand_Porsche', 'Brand_Renault', 'Brand_Skoda', 'Brand_Tata', 'Brand_Toyota', 'Brand_Volkswagen', 'Fuel_Type_Diesel', 'Fuel_Type_Petrol', 'Transmission_Manual', 'Owner_Type_Second', 'Owner_Type_Third', 'Age']]

y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

with open('rForest_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# new_data = [[0.09767, 15.80, 15.91, 13.74, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, 7]]

# new_data = [[21.000, 15.80, 15.91, 13.74, 6, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 7]]
# print(model.predict(new_data))
