import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Load the data
data = pd.read_csv("car_data.csv")

# Drop unnecessary columns
columns_to_drop = ["Name", "Location", "Power", "Seats"]
data.drop(columns=columns_to_drop, inplace=True)

# Convert categorical variables into one-hot encoded format
data = pd.get_dummies(data, columns=["Brand", "Fuel_Type", "Transmission", "Owner_Type"])

# Calculate the age of the car
current_year = 2023
data["Age"] = current_year - data["Year"]

# Feature scaling
scaler = StandardScaler()
data[['Kilometers_Driven', 'Mileage', 'Engine', 'New_Price', 'Age']] = scaler.fit_transform(data[['Kilometers_Driven', 'Mileage', 'Engine', 'New_Price', 'Age']])

# Define input features and target variable
X = data[['Kilometers_Driven', 'Mileage', 'Engine', 'New_Price', 'Age']]
y = data['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a neural network model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Output layer with a single neuron for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define callbacks for early stopping and model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_neural_network_model3.h5', save_best_only=True)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=8, callbacks=[early_stopping, model_checkpoint], verbose=1)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

print(X_test.head())

# Save the trained model
# model.save('neural_network_model.h5')






# # Drop unnecessary columns
# columns_to_drop = ["Name", "Location", "Power", "Seats"]
# data.drop(columns=columns_to_drop, inplace=True)

# # Define custom ordinal mappings
# brand_mapping = {
#     "Mercedes-Benz": 0, "BMW": 1, "Audi": 2, "Jaguar": 3, "Land": 4, "Porsche": 5, "Volvo": 6, "Toyota": 7,
#     "Honda": 8, "Volkswagen": 9, "Ford": 10, "Hyundai": 11, "Renault": 12, "Nissan": 13, "Mitsubishi": 14,
#     "Maruti": 15, "Mahindra": 16, "Tata": 17, "Skoda": 18, "Mini": 19, "Fiat": 20, "Isuzu": 21, "Jeep": 22, "Datsun": 23
# }

# fuel_type_mapping = {
#     "Fuel_Type_CNG": 0, "Fuel_Type_Diesel": 1, "Fuel_Type_Petrol": 2
# }

# owner_type_mapping = {
#     "Owner_Type_First": 0, "Owner_Type_Second": 1, "Owner_Type_Third": 2
# }

# transmission_mapping = {
#     "Transmission_Automatic": 0, "Transmission_Manual": 1
# }

# # Apply the custom ordinal mappings
# data["Brand_encoded"] = data["Brand"].map(brand_mapping)
# data["Fuel_Type_encoded"] = data["Fuel_Type"].map(fuel_type_mapping)
# data["Owner_Type_encoded"] = data["Owner_Type"].map(owner_type_mapping)
# data["Transmission_encoded"] = data["Transmission"].map(transmission_mapping)
