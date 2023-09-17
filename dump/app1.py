from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import datetime

model = open('pred_model.pkl','rb')
linear_regressor = pickle.load(model)

app = Flask(__name__)

def pred_linear(data):

    age = datetime.datetime.now().year - int(data['year'])

    km_driven = int(data['km-driven'])/1000

    if data['fuel'] == 'Petrol':
        petrol = True
        diesel = False
    else:
        petrol = False
        diesel = True

    manual = True if data['transmission'] == 'Manual' else False

    mileage = float(data['average'])

    engine = int(data['engine-cc'])/100

    new_price = float(data['new-price'])


    new_data = pd.DataFrame({
        'Kilometers_Driven': [km_driven],
        'Fuel_Type_Diesel': [diesel],
        'Fuel_Type_Petrol': [petrol],
        'Transmission_Manual': [manual],
        'Owner_Type_Second': [False],
        'Owner_Type_Third': [False],
        'Mileage': [mileage],
        'Engine': [engine],
        'Power': [121.3],
        'Seats': [5],
        'New_Price': [new_price],
        'Age': [age]
    })

    predicted_price = linear_regressor.predict(new_data)

    return f"{predicted_price[0]:.2f}"
    # return f"{predicted_price[0]:.2f} Lakhs"

    


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_price', methods=['POST'])
def check_price():
    data = request.form.to_dict()

    # message = pred_linear(data)
    linear = f"{pred_linear(data)} Lakhs"

    # return jsonify({'message': message})
    return jsonify({'linear': linear, 'greet': "Hello"})

if __name__ == '__main__':
    app.run(debug=True)
