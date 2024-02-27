import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import pickle

data = pd.read_csv("/Users/faizanhabib/Downloads/SeekCars-seekcars-v4 3/car_data.csv")

columns_to_drop = ["Name", "Location", "Power", "Seats"]
data.drop(columns=columns_to_drop, inplace=True)

data = pd.get_dummies(data, columns=["Brand", "Fuel_Type", "Transmission", "Owner_Type"])

current_year = 2024
data["Age"] = current_year - data["Year"]

data['Kilometers_Driven'] = data['Kilometers_Driven'] / 1000
data['Engine'] = data['Engine'] / 100

X = data[['Kilometers_Driven', 'Mileage', 'Engine', 'New_Price', 'Brand_Audi', 'Brand_BMW', 'Brand_Ford', 'Brand_Honda', 'Brand_Hyundai', 'Brand_Mahindra', 'Brand_Maruti', 'Brand_Mercedes', 'Brand_Nissan', 'Brand_Porsche', 'Brand_Renault', 'Brand_Skoda', 'Brand_Tata', 'Brand_Toyota', 'Brand_Volkswagen', 'Fuel_Type_Diesel', 'Fuel_Type_Petrol', 'Transmission_Manual', 'Owner_Type_Second', 'Owner_Type_Third', 'Age']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# param_grid = {
#     'C': [0.1, 1, 10],
#     'epsilon': [0.1, 0.01, 0.001],
#     'kernel': ['linear', 'rbf', 'poly']
# }

svm_regressor = SVR(kernel='linear')

svm_regressor.fit(X_train, y_train)

y_pred = svm_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

with open('svm_model.pkl', 'wb') as model_file:
    pickle.dump(svm_regressor, model_file)
