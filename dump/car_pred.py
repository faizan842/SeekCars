import pickle
import pandas as pd

model = open('pred_model.pkl','rb')
linear_regressor = pickle.load(model)

# price = linear_regressor.predict([[56.000,False,True,False,False,False,20.04,1650,103.3,5,32.04,6]])
new_data = pd.DataFrame({
    'Kilometers_Driven': [12.645],
    'Fuel_Type_Diesel': [False],
    'Fuel_Type_Petrol': [True],
    'Transmission_Manual': [False],
    'Owner_Type_Second': [False],
    'Owner_Type_Third': [False],
    'Mileage': [17],
    'Engine': [15.91],
    'Power': [121.3],
    'Seats': [5],
    'New_Price': [13.49],
    'Age': [4]
})

predicted_price = linear_regressor.predict(new_data)

print(f"Predicted Price: {predicted_price[0]:.2f} Lakhs")

# print(price)