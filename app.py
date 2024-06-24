from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el modelo entrenado
model = joblib.load('modelo_clima_regresion.pkl')
scaler = joblib.load('standard_scaler.pkl')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        avg_humidity = float(request.form['avg_humidity'])
        avg_dewpoint = float(request.form['avg_dewpoint'])
        min_pressure = float(request.form['min_pressure'])
        diff_pressure = float(request.form['diff_pressure'])
        
        # Escalar los datos de entrada
        scaled_data = scaler.transform([[avg_humidity, avg_dewpoint, min_pressure, diff_pressure]])
        
        # Crear un DataFrame con los datos escalados
        data_df = pd.DataFrame(scaled_data, columns=['Average humidity (%)', 'Average dewpoint (°F)', 'Minimum pressure', 'diff_pressure'])
        
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
