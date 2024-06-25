from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el modelo entrenado
model = joblib.load('modelo_clima_regresion_v2.pkl')
scaler = joblib.load('standard_scaler_v2.pkl')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        avg_humidity = float(request.form['avg_humidity'])
        avg_dewpoint = float(request.form['avg_dewpoint'])
        avg_barometer = float(request.form['avg_barometer'])
        max_pressure = float(request.form['max_pressure'])
        
        # Escalar los datos de entrada
        scaled_data = scaler.transform([[avg_humidity, avg_dewpoint, avg_barometer, max_pressure]])
        
        # Crear un DataFrame con los datos escalados
        data_df = pd.DataFrame(scaled_data, columns=['Average humidity (%)', 'Average dewpoint (°F)', 'Average barometer (in)', 'Maximum pressure'])
        
        # Imprimir el DataFrame para verificar los datos
        print("Datos recibidos (escalados):")
        print(data_df)
        
        # Realizar la predicción
        prediction = model.predict(data_df)[0]
        
        # Devolver la predicción como respuesta JSON
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
