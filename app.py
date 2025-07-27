# Para Spaces de Hugging Face, la app principal debe llamarse app.py
# Este archivo es una copia de ejemplo_2_.py

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, pipeline

@st.cache_resource
def load_models():
    # FLAN-T5 para QA
    flan_name = "google/flan-t5-small"
    flan_tokenizer = AutoTokenizer.from_pretrained(flan_name)
    flan_model = AutoModelForSeq2SeqLM.from_pretrained(flan_name)

    # DistilBERT para clasificación (modelo base)
    distil_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_analyzer = pipeline("sentiment-analysis", model=distil_name)

    # DistilBERT fine-tuned propio
    custom_name = "juancmamacias/jd-jcms"
    custom_analyzer = pipeline("sentiment-analysis", model=custom_name)

    return flan_tokenizer, flan_model, sentiment_analyzer, custom_analyzer



# Cargar modelos antes de cualquier uso
st.set_page_config(page_title="SLM Demo: QA + Sentiment", page_icon="🧠", layout="wide")

# Ocultar header, footer y menú de Streamlit
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
flan_tokenizer, flan_model, sentiment_analyzer, custom_analyzer = load_models()

# Layout con dos columnas
col1, col2 = st.columns([1,2])


with col1:
    st.title("🧠 Small Language Models Demo")
    st.markdown("""
    Esta app compara tres Small Language Models:
    - `flan-t5-small` para responder preguntas.
    - `distilBERT` para análisis de sentimiento.
    - `distilBERT` fine-tuned para análisis de sentimiento.
    """)
    st.markdown("""
    ## Autores
    - Juan Domingo ([GitHub](https://github.com/jdomdev))
    - Juan Carlos Macías ([GitHub](https://github.com/juancmacias))
    """)

with col2:
    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.text_input("💬 Escribe una pregunta o frase para analizar:")

    if st.button("Procesar") and question:
        with st.spinner("Procesando..."):
            # ➤ Respuesta con FLAN-T5 usando prompt explícito
            prompt = f"Answer the following question: {question}"
            input_ids = flan_tokenizer(prompt, return_tensors="pt").input_ids
            outputs = flan_model.generate(input_ids, max_length=50)
            flan_answer = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # ➤ Clasificación con DistilBERT base
            sentiment = sentiment_analyzer(question)[0]
            sentiment_label = sentiment['label']
            sentiment_score = round(sentiment['score'], 3)

            # Traducción de etiquetas para ambos modelos
            label_map = {"LABEL_0": "NEGATIVO", "LABEL_1": "POSITIVO", "NEGATIVE": "NEGATIVO", "POSITIVE": "POSITIVO"}
            sentiment_label = label_map.get(sentiment_label, sentiment_label)

            # ➤ Clasificación con DistilBERT fine-tuned propio
            custom_sentiment = custom_analyzer(question)[0]
            custom_label = custom_sentiment['label']
            custom_score = round(custom_sentiment['score'], 3)
            custom_label = label_map.get(custom_label, custom_label)

            # Guardar en historial
            st.session_state.history.append({
                "question": question,
                "answer": flan_answer,
                "sentiment": f"{sentiment_label} ({sentiment_score})",
                "custom_sentiment": f"{custom_label} ({custom_score})"
            })

    # Mostrar historial
    if st.session_state.history:
        st.markdown("### 📜 Historial")
        for i, item in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"""
            **{i}. Entrada:** {item['question']}  
            🧠 **Respuesta (FLAN):** {item['answer']}  
            ❤️ **Sentimiento (base):** {item['sentiment']}  
            💙 **Sentimiento (propio):** {item['custom_sentiment']}  
            ---""")
