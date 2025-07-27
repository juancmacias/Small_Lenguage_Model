import streamlit as st
import os
import json
import requests
from transformers import pipeline
from dotenv import load_dotenv # Importar load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# --- 1. Configuración de la Ruta del Modelo Local ---
# Construye la ruta absoluta al directorio del modelo basándose en la ubicación del script.
# Esto hace que la aplicación sea más robusta y no dependa del directorio de trabajo actual.
try:
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError:
    # Si __file__ no está definido (por ejemplo, en un entorno interactivo como un notebook)
    SCRIPT_DIR = os.getcwd()

MODEL_PATH = os.path.join(SCRIPT_DIR, "fine_tuned_sentiment_model_full_data")

# --- 2. Carga del Modelo Local (SLM) ---
@st.cache_resource
def load_local_pipeline():
    """Carga el pipeline de análisis de sentimiento del modelo local."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"El directorio del modelo '{MODEL_PATH}' no se encuentra.")
        return None, None
    
    try:
        # Carga el pipeline de análisis de sentimiento
        sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_PATH, tokenizer=MODEL_PATH)
        
        # Carga el mapeo de etiquetas desde el archivo de configuración
        config_path = os.path.join(MODEL_PATH, 'config.json')
        with open(config_path) as f:
            config = json.load(f)
        
        id2label = config.get('id2label')
        if id2label is None:
            # Si id2label no está en el config, lo creamos manualmente
            id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        else:
            id2label = {int(k): v for k, v in id2label.items()}

        return sentiment_pipeline, id2label
    except Exception as e:
        st.error(f"Error al cargar el modelo local: {e}")
        return None, None

slm_pipeline, id2label_map = load_local_pipeline()

def query_local_model(review):
    """Realiza una predicción de sentimiento usando el SLM local."""
    if slm_pipeline is None or id2label_map is None:
        return "Error: El modelo local no está disponible."

    try:
        # Realiza la predicción
        result = slm_pipeline(review)[0]
        label_id = int(result['label'].split('_')[1])
        label = id2label_map.get(label_id, "Desconocido")
        score = result['score']
        
        return f"Resultado: {label} ({score:.2%})"
    except Exception as e:
        return f"Error durante la predicción: {e}"

# --- 3. Lógica para el LLM (Hugging Face Inference API) ---
# LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"
# LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct" # Nuevo modelo Llama 3
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

HF_API_URL = f"https://api-inference.huggingface.co/models/{LLM_MODEL}"

# Obtener el token de las variables de entorno
HF_TOKEN = os.environ.get("HF_TOKEN")

def query_llm_api(review):
    """Consulta un LLM a través de la API de Inferencia de Hugging Face usando requests.post."""
    if not HF_TOKEN:
        return "Error: No se encontró el token de Hugging Face. Asegúrate de que está en tu archivo .env"

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    # Prompt Engineering: Le damos al LLM una instrucción clara y concisa.
    # Usamos el formato de mensajes esperado por modelos de instrucción/conversación.
    messages = [
        {"role": "system", "content": "Analyze the sentiment of the following movie review. Respond only with the word 'POSITIVE' or 'NEGATIVE'."},
        {"role": "user", "content": review}
    ]
    payload = {
    "inputs": [
        {"role": "system", "content": "Classify the sentiment of the following movie review as POSITIVE or NEGATIVE."},
        {"role": "user", "content": review}
    ],
    "parameters": {
        "temperature": 0.1,
        "max_new_tokens": 20,
        "return_full_text": False
    }
}

    # payload = {
    #     "inputs": messages,
    #     "parameters": {
    #         "max_new_tokens": 10,
    #         "temperature": 0.1,
    #         "return_full_text": False, # Solo queremos la respuesta del modelo
    #     }
    # }
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60) # Aumentamos el timeout
        response.raise_for_status()  # Lanza una excepción para códigos de estado 4xx/5xx
        
        result = response.json()
        # La respuesta para modelos de chat suele ser un array con un diccionario que contiene 'generated_text'
        if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
            answer = result[0]['generated_text'].strip()
            return f"Resultado: {answer}"
        else:
            return f"Error: Formato de respuesta inesperado de la API. Respuesta: {result}"

    except requests.exceptions.RequestException as e:
        # Captura errores específicos de la solicitud HTTP
        if "401 Client Error" in str(e) or "403 Client Error" in str(e):
            return f"Error de autenticación o acceso denegado al LLM. Asegúrate de que tu token es válido y has aceptado los términos del modelo {LLM_MODEL} en Hugging Face. Error: {e}"
        elif "404 Client Error" in str(e):
            return f"Error: Modelo {LLM_MODEL} no encontrado o no disponible para inferencia. Error: {e}"
        elif "503 Service Unavailable" in str(e):
            return f"Error: El modelo {LLM_MODEL} está cargando o no está disponible temporalmente. Intenta de nuevo en unos segundos. Error: {e}"
        return f"Error de conexión con la API: {e}"
    except Exception as e:
        return f"Error al procesar la respuesta de la API: {e}"

# --- 4. Interfaz de Streamlit ---
st.set_page_config(layout="wide")
st.title("Comparador de Modelos de Sentimiento: SLM vs LLM")

st.info("""
Introduce una reseña de película en inglés para comparar el rendimiento de un modelo pequeño y eficiente (DistilBERT, local) 
contra un modelo de lenguaje grande (Llama 3 8B Instruct, en la nube).
""")

user_input = st.text_area(
    "Introduce la reseña aquí:",
    "This movie was absolutely fantastic, a true masterpiece of cinema!",
    height=150
)

if st.button("Analizar Sentimiento", type="primary"):
    if not user_input:
        st.warning("Por favor, introduce una reseña.")
    elif slm_pipeline is None:
        st.error("La aplicación no puede iniciarse porque el modelo local no se pudo cargar.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tu Modelo (DistilBERT - Local)")
            with st.spinner("Procesando con el modelo local..."):
                slm_result = query_local_model(user_input)
                st.markdown(f"**Análisis:**\n{slm_result}")

        with col2:
            st.subheader("LLM (Llama 3 8B Instruct - Hugging Face API)")
            with st.spinner("Consultando al LLM en la nube..."):
                llm_result = query_llm_api(user_input)
                st.markdown(f"**Análisis:**\n{llm_result}")