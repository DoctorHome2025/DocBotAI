import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import json

# --- Definir ruta para el modelo guardado ---
MODEL_PATH = 'modelo_medico.pkl'
SINTOMAS_PATH = 'sintomas.json'
CLASSES_PATH = 'clases.json'

# --- Función para cargar o entrenar el modelo ---
def cargar_o_entrenar_modelo():
    global todos_sintomas, modelo, clases
    
    # Comprobar si el modelo ya está guardado
    if os.path.exists(MODEL_PATH) and os.path.exists(SINTOMAS_PATH) and os.path.exists(CLASSES_PATH):
        print("Cargando modelo y síntomas desde archivo...")
        modelo = pickle.load(open(MODEL_PATH, 'rb'))
        todos_sintomas = json.load(open(SINTOMAS_PATH, 'r'))
        clases = json.load(open(CLASSES_PATH, 'r'))
    else:
        print("Entrenando nuevo modelo...")
        # --- Cargar datos ---
        df = pd.read_excel("enf2025.xlsx")
        
        # --- Preprocesamiento de síntomas ---
        df['Sintomas'] = df['Sintomas'].str.lower().fillna('')
        df['Sintomas'] = df['Sintomas'].apply(lambda x: [s.strip() for s in x.split(',')])
        
        # --- Extraer síntomas únicos ---
        todos_sintomas = sorted(set(s for lista in df['Sintomas'] for s in lista))
        print(f"Total de síntomas en el dataset: {len(todos_sintomas)}")
        
        # --- Convertir lista de síntomas a vector binario ---
        X = np.array([[1 if s in sintoma_lista else 0 for s in todos_sintomas] for sintoma_lista in df['Sintomas']])
        y = df['Enfermedad'].values
        
        # --- Crear y entrenar modelo ---
        modelo = RandomForestClassifier(n_estimators=50, random_state=42)
        modelo.fit(X, y)
        
        # Guardar las clases (enfermedades)
        clases = modelo.classes_.tolist()
        
        # --- Guardar modelo, síntomas y clases ---
        pickle.dump(modelo, open(MODEL_PATH, 'wb'))
        json.dump(todos_sintomas, open(SINTOMAS_PATH, 'w'))
        json.dump(clases, open(CLASSES_PATH, 'w'))
    
    return todos_sintomas, modelo, clases

# Cargar modelo al inicio
todos_sintomas, modelo, clases = cargar_o_entrenar_modelo()

# --- Función para predecir enfermedad desde síntomas ---
def predecir_enfermedad(sintomas_usuario):
    # Asegurarse de que los síntomas están en minúsculas y sin espacios extras
    sintomas_usuario = [s.lower().strip() for s in sintomas_usuario]
    
    # Verificar que haya síntomas válidos
    if not sintomas_usuario:
        return "No se han seleccionado síntomas", 0.0
    
    # Crear vector de síntomas
    vector = np.array([[1 if s in sintomas_usuario else 0 for s in todos_sintomas]])
    
    # Hacer predicción
    pred_proba = modelo.predict_proba(vector)[0]
    idx = np.argmax(pred_proba)
    enfermedad = clases[idx]
    probabilidad = pred_proba[idx] * 100
    
    return enfermedad, probabilidad

# --- Función para obtener todos los síntomas disponibles ---
def get_todos_sintomas():
    return todos_sintomas

# --- Función para extraer síntomas del mensaje ---
def extraer_sintomas_de_mensaje(mensaje):
    mensaje = mensaje.lower()
    sintomas_detectados = []
    
    # Solo detectar síntomas exactos de la lista
    for sintoma in todos_sintomas:
        if sintoma in mensaje:
            sintomas_detectados.append(sintoma)
    
    print(f"Síntomas detectados en mensaje: {sintomas_detectados}")    
    return sintomas_detectados

# --- Función para generar respuesta del chatbot ---
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