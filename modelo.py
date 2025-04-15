import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical # type: ignore

# --- Cargar datos ---
df = pd.read_excel("enf2025.xlsx")

# --- Preprocesamiento de síntomas ---
df['Sintomas'] = df['Sintomas'].str.lower().fillna('')
df['Sintomas'] = df['Sintomas'].apply(lambda x: [s.strip() for s in x.split(',')])

# --- Extraer síntomas únicos ---
todos_sintomas = sorted(set(s for lista in df['Sintomas'] for s in lista))
print(f"Total de síntomas en el dataset: {len(todos_sintomas)}")

# --- Convertir lista de síntomas a vector binario ---
def sintomas_a_vector(sintoma_lista):
    return [1 if s in sintoma_lista else 0 for s in todos_sintomas]

# --- Preparar datos de entrenamiento ---
X = np.array([sintomas_a_vector(s) for s in df['Sintomas']])
y_etiquetas = df['Enfermedad'].values

# --- Codificación de etiquetas (enfermedades) ---
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_etiquetas)
y = to_categorical(y_encoded)

# --- Crear y entrenar modelo ---
modelo = Sequential([
    Dense(32, activation='relu', input_shape=(len(todos_sintomas),)),
    Dense(16, activation='relu'),
    Dense(len(set(y_etiquetas)), activation='softmax')
])

modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
modelo.fit(X, y, epochs=200, verbose=0)  # Entrenamiento silencioso

# --- Función para predecir enfermedad desde síntomas ---
def predecir_enfermedad(sintomas_usuario):
    # Asegurarse de que los síntomas están en minúsculas y sin espacios extras
    sintomas_usuario = [s.lower().strip() for s in sintomas_usuario]
    
    # Verificar que haya síntomas válidos
    if not sintomas_usuario:
        return "No se han seleccionado síntomas", 0.0
    
    # Crear vector de síntomas
    vector = np.array([1 if s in sintomas_usuario else 0 for s in todos_sintomas])
    
    # Hacer predicción
    pred = modelo.predict(np.array([vector]), verbose=0)
    idx = np.argmax(pred)
    enfermedad = encoder.inverse_transform([idx])[0]
    probabilidad = pred[0][idx] * 100
    
    return enfermedad, probabilidad

# --- Función para obtener todos los síntomas disponibles ---
def get_todos_sintomas():
    return todos_sintomas

# --- Funciones para el chatbot ---
# Función para extraer síntomas exactos del mensaje del usuario
def extraer_sintomas_de_mensaje(mensaje):
    mensaje = mensaje.lower()
    sintomas_detectados = []
    
    # Solo detectar síntomas exactos de la lista
    for sintoma in todos_sintomas:
        if sintoma in mensaje:
            sintomas_detectados.append(sintoma)
    
    print(f"Síntomas detectados en mensaje: {sintomas_detectados}")    
    return sintomas_detectados

# Función para generar una respuesta del chatbot
def generar_respuesta_chatbot(mensaje_usuario):
    # Extraer síntomas del mensaje
    sintomas_detectados = extraer_sintomas_de_mensaje(mensaje_usuario)
    
    # Si no hay síntomas detectados
    if not sintomas_detectados:
        return "No he podido identificar síntomas en tu mensaje. Por favor, menciona tus síntomas de forma más clara, o prueba usando la opción de selección de síntomas."
    
    # Predecir enfermedad
    enfermedad, probabilidad = predecir_enfermedad(sintomas_detectados)
    
    # Formar respuesta
    respuesta = f"Según los síntomas que mencionas ({', '.join(sintomas_detectados)}), "
    respuesta += f"es posible que tengas {enfermedad} con una probabilidad del {probabilidad:.2f}%. "
    respuesta += "Recuerda que esto es solo una orientación y debes consultar a un médico para un diagnóstico profesional."
    
    return respuesta