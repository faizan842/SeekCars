import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler

# Load the data
data = pd.read_csv("car_data.csv")

# Drop unnecessary columns
columns_to_drop = ["Name", "Location", "Power", "Seats"]
data.drop(columns=columns_to_drop, inplace=True)

# Convert categorical variables into one-hot encoded format
# (You can also use pd.get_dummies for this)
categorical_columns = ["Brand", "Fuel_Type", "Owner_Type", "Transmission"]
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Calculate the age of the car
current_year = 2023
data["Age"] = current_year - data["Year"]

# Feature scaling using Min-Max scaling (normalize between 0 and 1)
scaler = MinMaxScaler()
data[['Kilometers_Driven', 'Mileage', 'Engine', 'New_Price', 'Age']] = scaler.fit_transform(data[['Kilometers_Driven', 'Mileage', 'Engine', 'New_Price', 'Age']])

# Feature engineering: Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(data[['Kilometers_Driven', 'Mileage', 'Engine', 'New_Price', 'Age']])
X_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names(['Kilometers_Driven', 'Mileage', 'Engine', 'New_Price', 'Age']))
data = pd.concat([data, X_poly], axis=1)

# Define input features and target variable
X = data.drop(columns=['Price'])
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
model_checkpoint = ModelCheckpoint('best_neural_network_model.h5', save_best_only=True)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=8, callbacks=[early_stopping, model_checkpoint], verbose=1)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Save the trained model
model.save('neural_network_model.h5')
