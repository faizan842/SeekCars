import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
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
#     'n_neighbors': [3, 5, 7, 9],
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidean', 'manhattan']
# }

knn_regressor = KNeighborsRegressor(n_neighbors=9,weights='uniform',metric='manhattan')

# grid_search = GridSearchCV(estimator=knn_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(X_train, y_train)

# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_

knn_regressor.fit(X_train, y_train)

y_pred = knn_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

with open('knn_model.pkl', 'wb') as model_file:
    pickle.dump(knn_regressor, model_file)
