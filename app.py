from flask import Flask, render_template, request, jsonify
import modelo
import os
import json
import time

app = Flask(__name__)

# Obtener todos los síntomas desde el modelo
sintomas_disponibles = modelo.get_todos_sintomas()
print(f"Cargando aplicación con {len(sintomas_disponibles)} síntomas disponibles")

# Log de interacciones
LOG_PATH = 'interacciones.json'

def guardar_interaccion(mensaje, sintomas=None, enfermedad=None):
    """Guarda la interacción en el archivo de registro"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    nueva_interaccion = {
        "timestamp": timestamp,
        "mensaje": mensaje,
        "sintomas": sintomas if sintomas else [],
        "enfermedad": enfermedad
    }
    
    # Cargar registros existentes o crear uno nuevo
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, 'r') as f:
                interacciones = json.load(f)
        except json.JSONDecodeError:
            interacciones = []
    else:
        interacciones = []
    
    # Añadir nueva interacción y guardar
    interacciones.append(nueva_interaccion)
    with open(LOG_PATH, 'w') as f:
        json.dump(interacciones, f, indent=2)

# Ruta principal que muestra el formulario
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', sintomas=sintomas_disponibles)

# Ruta de verificación de salud
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

# Ruta para diagnóstico con síntomas seleccionados
@app.route('/diagnosticar', methods=['POST'])
def diagnosticar():
    sintomas_seleccionados = request.form.getlist('sintomas')
    print(f"Síntomas seleccionados: {sintomas_seleccionados}")
    
    if sintomas_seleccionados:
        enfermedad, probabilidad = modelo.predecir_enfermedad(sintomas_seleccionados)
        resultado = f"{enfermedad} (Probabilidad: {probabilidad:.2f}%)"
        guardar_interaccion("diagnostico-web", sintomas_seleccionados, enfermedad)
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
    
    # Registrar mensaje
    guardar_interaccion(mensaje_usuario)
    
    # Generar respuesta del chatbot
    respuesta = modelo.generar_respuesta_chatbot(mensaje_usuario)
    
    # Si se detectaron síntomas, actualizar la interacción
    sintomas_detectados = modelo.extraer_sintomas_de_mensaje(mensaje_usuario)
    if sintomas_detectados:
        enfermedad, _ = modelo.predecir_enfermedad(sintomas_detectados)
        guardar_interaccion(mensaje_usuario, sintomas_detectados, enfermedad)
    
    return jsonify({'respuesta': respuesta})

# Ejecutar la app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)