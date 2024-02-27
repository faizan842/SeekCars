import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_train_scaled)

train_cluster_labels = kmeans.predict(X_train_scaled)
test_cluster_labels = kmeans.predict(X_test_scaled)

with open('kmeans_model.pkl', 'wb') as file:
    pickle.dump(kmeans, file)

regression_models = {}
for cluster_label in range(n_clusters):
    X_train_cluster = X_train[train_cluster_labels == cluster_label]
    y_train_cluster = y_train[train_cluster_labels == cluster_label]
    regression_model = RandomForestRegressor(n_estimators=100, random_state=42)
    regression_model.fit(X_train_cluster, y_train_cluster)
    regression_models[cluster_label] = regression_model

y_pred = []
for i, cluster_label in enumerate(test_cluster_labels):
    regression_model = regression_models[cluster_label]
    y_pred_cluster = regression_model.predict([X_test.iloc[i]])
    y_pred.append(y_pred_cluster[0])

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)
