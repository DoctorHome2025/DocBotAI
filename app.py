from flask import Flask, render_template, request, jsonify
import modelo
import os

app = Flask(__name__)

# Obtener todos los síntomas desde el modelo
sintomas_disponibles = modelo.get_todos_sintomas()
print(f"Cargando aplicación con {len(sintomas_disponibles)} síntomas disponibles")

# Ruta principal que muestra el formulario
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', sintomas=sintomas_disponibles)

# Ruta para diagnóstico con síntomas seleccionados
@app.route('/diagnosticar', methods=['POST'])
def diagnosticar():
    sintomas_seleccionados = request.form.getlist('sintomas')
    print(f"Síntomas seleccionados: {sintomas_seleccionados}")
    
    if sintomas_seleccionados:
        enfermedad, probabilidad = modelo.predecir_enfermedad(sintomas_seleccionados)
        resultado = f"{enfermedad} (Probabilidad: {probabilidad:.2f}%)"
    else:
        resultado = "Por favor selecciona al menos un síntoma."
    
    return jsonify({'resultado': resultado})

# Ruta para el chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    datos = request.get_json()
    mensaje_usuario = datos.get('mensaje', '')
    print(f"Mensaje del usuario: {mensaje_usuario}")
    
    if not mensaje_usuario:
        return jsonify({'respuesta': 'No he recibido ningún mensaje.'})
    
    # Generar respuesta del chatbot
    respuesta = modelo.generar_respuesta_chatbot(mensaje_usuario)
    return jsonify({'respuesta': respuesta})

# Ejecutar la app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
