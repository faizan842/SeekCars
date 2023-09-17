from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import datetime



app = Flask(__name__)

def pred_linear(new_data):
    model = open('models/linear_model.pkl','rb')
    linear_regressor = pickle.load(model)

    predicted_price = linear_regressor.predict(new_data)
    return f"{predicted_price[0]:.2f}"

def pred_rForest(new_data):

    model = open('models/rForest_model.pkl','rb')
    rForest = pickle.load(model)

    res = rForest.predict(new_data)
    return f"{res[0]:.2f}"

def pred_dTree(new_data):
    model = open('models/dTree_model.pkl','rb')
    dTree = pickle.load(model)

    res = dTree.predict(new_data)
    return f"{res[0]:.2f}"

def pred_xgBoost(new_data):
    model = open('models/xgboost.pkl','rb')
    xgBoost = pickle.load(model)

    res = xgBoost.predict(new_data)
    return f"{res[0]:.2f}"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_price', methods=['POST'])
def check_price():
    data = request.form.to_dict()

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

    audi = bmw =  ford =  honda =  hyundai = mahindra = maruti = mercedes = nissan = porsche = renault = skoda = tata = toyota = volkswagen = False
   
    if data['brand-name'] == "Audi":
        audi = True
    elif data['brand-name'] == "BMW":
        bmw = True 
    elif data['brand-name'] == "Ford":
        ford = True 
    elif data['brand-name'] == "Honda":
        honda = True 
    elif data['brand-name'] == "Hyundai":
        hyundai = True 
    elif data['brand-name'] == "Mahindra":
        mahindra = True 
    elif data['brand-name'] == "Maruti":
        maruti = True 
    elif data['brand-name'] == "Mercedes":
        mercedes = True 
    elif data['brand-name'] == "Nissan":
        nissan = True 
    elif data['brand-name'] == "Porsche":
        porsche = True 
    elif data['brand-name'] == "Renault":
        renault = True 
    elif data['brand-name'] == "Skoda":
        skoda = True 
    elif data['brand-name'] == "Tata":
        tata = True 
    elif data['brand-name'] == "Toyota":
        toyota = True 
    elif data['brand-name'] == "Volkswagen":
        volkswagen = True 
    else:
        hyundai = True 

    new_data = pd.DataFrame({
        'Kilometers_Driven': [km_driven],
        'Mileage': [mileage],
        'Engine': [engine],
        'New_Price': [new_price],
        'Brand_Audi': [audi],
        'Brand_BMW': [bmw],
        'Brand_Ford': [ford],
        'Brand_Honda': [honda],
        'Brand_Hyundai': [hyundai],
        'Brand_Mahindra': [mahindra],
        'Brand_Maruti': [maruti],
        'Brand_Mercedes': [mercedes],
        'Brand_Nissan': [nissan],
        'Brand_Porsche': [porsche],
        'Brand_Renault': [renault],
        'Brand_Skoda': [skoda],
        'Brand_Tata': [tata],
        'Brand_Toyota': [toyota],
        'Brand_Volkswagen': [volkswagen],
        'Fuel_Type_Diesel': [diesel],
        'Fuel_Type_Petrol': [petrol],
        'Transmission_Manual': [manual],
        'Owner_Type_Second': [False],
        'Owner_Type_Third': [False],
        'Age' : [age]
    })

    linear = f"{pred_linear(new_data)} Lakhs"

    rf = f"{pred_rForest(new_data)} lakhs"

    dT = f"{pred_dTree(new_data)} lakhs"

    xg = f"{pred_xgBoost(new_data)} lakhs"

    if data["model-name"] == "1":
        return jsonify({'result':linear, 'arr': {'Random Forest': rf, 'Decision Tree': dT, 'Linear': linear, 'XgBoost': xg}})
    elif data["model-name"] == "2":
        return jsonify({'result':rf, 'arr': {'Random Forest': rf, 'Decision Tree': dT, 'Linear': linear, 'XgBoost': xg}})
    elif data["model-name"] == "3":
        return jsonify({'result':dT, 'arr': {'Random Forest': rf, 'Decision Tree': dT, 'Linear': linear, 'XgBoost': xg}})
    elif data["model-name"] == "4":
        return jsonify({'result':xg, 'arr': {'Random Forest': rf, 'Decision Tree': dT, 'Linear': linear, 'XgBoost': xg}})
    else:
        return jsonify({'result':linear, 'arr': {'Random Forest': rf, 'Decision Tree': dT, 'Linear': linear, 'XgBoost': xg}})

    

if __name__ == '__main__':
    app.run(debug=True)
