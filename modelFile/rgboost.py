import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import pickle

# Load the dataset
data = pd.read_csv("car_data.csv")

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

data['Kilometers_Driven'] = data['Kilometers_Driven'] / 1000

data['Engine'] = data['Engine'] / 100

columns_to_drop = ["Name", "Power","Seats","Model Name"]
data.drop(columns=columns_to_drop, inplace=True)

data = data.astype(float)
print(data.head())

data.to_csv('sorted_data.csv', index=False)

X = data[['Kilometers_Driven', 'Mileage', 'Engine', 'New_Price', 'Location', 'Brand_Audi', 'Brand_BMW', 'Brand_Ford', 'Brand_Honda', 'Brand_Hyundai', 'Brand_Mahindra', 'Brand_Maruti', 'Brand_Mercedes', 'Brand_Nissan', 'Brand_Porsche', 'Brand_Renault', 'Brand_Skoda', 'Brand_Tata', 'Brand_Toyota', 'Brand_Volkswagen', 'Fuel_Type_Diesel', 'Fuel_Type_Petrol', 'Transmission_Manual', 'Owner_Type_Second', 'Owner_Type_Third', 'Age']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

gb_regressor = GradientBoostingRegressor(random_state=42)

grid_search = GridSearchCV(estimator=gb_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

best_gb_regressor = grid_search.best_estimator_
best_gb_regressor.fit(X_train, y_train)

y_pred = best_gb_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Save the trained model
with open('xgboost.pkl', 'wb') as model_file:
    pickle.dump(best_gb_regressor, model_file)
