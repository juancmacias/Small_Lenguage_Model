import streamlit as st
import os
import json
from dotenv import load_dotenv
from transformers import pipeline
from huggingface_hub import InferenceClient

# Cargar variables de entorno
load_dotenv()

# --- 1. Configuración de la Ruta del Modelo Local ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

MODEL_PATH = os.path.join(SCRIPT_DIR, "fine_tuned_sentiment_model_full_data")
# Cargar el token de Hugging Face
HF_TOKEN = os.environ.get("HF_TOKEN")

# --- 2. Carga del Modelo Local (SLM) ---
@st.cache_resource
def load_local_pipeline():
    if not os.path.exists(MODEL_PATH):
        st.error(f"El directorio del modelo '{MODEL_PATH}' no se encuentra.")
        return None, None
    
    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_PATH, tokenizer=MODEL_PATH)
        config_path = os.path.join(MODEL_PATH, 'config.json')
        with open(config_path) as f:
            config = json.load(f)
        id2label = config.get('id2label')
        if id2label is None:
            id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        else:
            id2label = {int(k): v for k, v in id2label.items()}
        return sentiment_pipeline, id2label
    except Exception as e:
        st.error(f"Error al cargar el modelo local: {e}")
        return None, None

slm_pipeline, id2label_map = load_local_pipeline()

def query_local_model(review):
    if slm_pipeline is None or id2label_map is None:
        return "Error: El modelo local no está disponible."

    try:
        result = slm_pipeline(review)[0]
        label_id = int(result['label'].split('_')[1])
        label = id2label_map.get(label_id, "Desconocido")
        score = result['score']
        return f"Resultado: {label} ({score:.2%})"
    except Exception as e:
        return f"Error durante la predicción: {e}"

# --- 3. LLM vía InferenceClient ---
llm_model_info = """
| Modelo                                             | Tamaño   | Instrucción / Chat | Arquitectura Base  | Destacado por |
|----------------------------------------------------|----------|--------------------|--------------------|---------------|
| meta-llama/Meta-Llama-3-8B-Instruct                | 8B       | Sí (Instruct)      | LLaMA 3            | Calidad alta en tareas de reasoning, uso general |
| HuggingFaceH4/zephyr-7b-beta                       | 7B       | Sí (Chat)          | Mistral-like       | Fluidez conversacional y rendimiento notable con prompts |
| mistralai/Mistral-7B-Instruct-v0.2                 | 7B       | Sí (Instruct)      | Mistral            | Precisión y velocidad, muy versátil en tareas NLP |
| Qwen/Qwen1.5-7B-Chat                               | 7B       | Sí (Chat)          | Qwen               | Buen rendimiento multilingüe y contextualización |
| NousResearch/Nous-Hermes-2-Mistral-7B-DPO          | 7B       | Sí (Chat/Instruct) | Mistral            | Ajustado con DPO, enfoque en calidad de respuestas |
| openchat/openchat-3.5-0106                         | ~7B      | Sí (Chat)          | Mixtral/OpenChat   | Fine-tuned para alineación, estilo ChatGPT 3.5 |
| google/gemma-2b-it                                 | 2B       | Sí (Instruct)      | Gemma              | Ligero y eficiente, ideal para recursos limitados |
| lmsys/vicuna-7b-v1.5                               | 7B       | Sí (Chat)          | LLaMA 1 finetuned  | Alineado con diálogo humano, estilo GPT-like |
| OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5     | 12B      | Sí (Chat)          | Pythia             | Entrenado colaborativamente, énfasis en transparencia |
| stabilityai/stablelm-tuned-alpha-3b                | 3B       | Sí (Chat-like)     | StableLM           | Pequeño y rápido, código abierto accesible |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0                 | 1.1B     | Sí (Chat)          | TinyLlama          | Ultra compacto, ideal para dispositivos locales |
| tiiuae/falcon-7b-instruct                          | 7B       | Sí (Instruct)      | Falcon             | Rendimiento robusto, fuerte comprensión de tareas |
| databricks/dolly-v2-3b                             | 3B       | Sí (Instruct)      | GPT-J              | Simplicidad y fine-tuning open-source para instruct |
"""

LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

@st.cache_resource
def get_inference_client():
    if not HF_TOKEN:
        st.error("No se encontró el token de Hugging Face. Asegúrate de tener un archivo .env con HF_TOKEN.")
        return None
    return InferenceClient(model=LLM_MODEL, token=HF_TOKEN)

client = get_inference_client()

def query_llm_api(review):
    if client is None:
        return "Error: No se pudo inicializar el cliente de Hugging Face."

    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": "Analyze the sentiment. Respond with POSITIVE or NEGATIVE."},
                {"role": "user", "content": review}
            ],
            max_tokens=10,
            temperature=0.1,
        )

        return f"Resultado: {response.choices[0].message['content'].strip()}"
    except Exception as e:
        return f"Error al consultar el LLM: {e}"

# --- 4. Interfaz de Streamlit ---
st.set_page_config(layout="wide")
st.title("Comparador de Modelos de Sentimiento: SLM vs LLM")

st.info(f"""
Introduce una reseña de película en inglés para comparar el rendimiento de un modelo pequeño y eficiente (DistilBERT, local)  
contra un modelo de lenguaje grande LLM **{LLM_MODEL}**, en la nube.
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
                st.markdown(f"{slm_result}")

        with col2:
            st.subheader(f"LLM ({LLM_MODEL.split('/')[1]} - Hugging Face API)")
            with st.spinner("Consultando al LLM en la nube..."):
                llm_result = query_llm_api(user_input)
                st.markdown(f"{llm_result}")

# st.markdown(f"### Modelos LLM recomendados:\n{llm_model_info}")                
        