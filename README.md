
# Detección sentimiento con SLM (DistilBERT, FLAN-T5)

Este proyecto muestra cómo entrenar, probar y desplegar modelos de lenguaje pequeños (SLM) para análisis de sentimiento y respuesta a preguntas usando Hugging Face Transformers, PyTorch y Streamlit. Incluye una app interactiva lista para Hugging Face Spaces.


## Requisitos

- Python 3.8 o superior
- Acceso a internet para descargar modelos preentrenados
- Para la app web: Streamlit


## Instalación de dependencias

Instala todas las dependencias necesarias ejecutando:

```bash
pip install -r requirements.txt
```

El archivo `requirements.txt` incluye:
- transformers[torch]
- torch
- streamlit
- datasets, evaluate, scikit-learn, accelerate


## Ejecución paso a paso (entrenamiento y prueba local)

1. **Clona o descarga este repositorio y entra en la carpeta del proyecto.**
2. **(Opcional) Crea y activa un entorno virtual:**
   - **Windows:**
     ```powershell
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - **Linux/Mac:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
3. **Instala las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Entrena y prueba el modelo:**
   ```bash
   python train_sentiment_model.py
   ```
5. **Ejecuta la app Streamlit localmente:**
   ```bash
   streamlit run app.py
   ```


## Despliegue en Hugging Face Spaces

Puedes desplegar la app web en Hugging Face Spaces siguiendo estos pasos:

1. Sube los archivos `app.py` y `requirements.txt` a tu nuevo Space (tipo Streamlit) en Hugging Face.
2. (Opcional) Añade este `README.md` para documentar tu Space.
3. El Space detectará automáticamente `app.py` y lanzará la app.

**Nota:** Si tu modelo fine-tuned es privado, asegúrate de que el Space tenga acceso o hazlo público.

### Ejecución local rápida

```bash
streamlit run app.py
```


## Notas
- El script `train_sentiment_model.py` entrena y evalúa el modelo fine-tuned.
- El archivo `app.py` es la app web para Spaces y para ejecución local.
- Si tienes problemas con dependencias, asegúrate de tener la última versión de pip: `pip install --upgrade pip`.
- El entrenamiento es rápido porque el dataset es pequeño y el modelo es ligero.

---


---

**Autores:**
- Juan Domingo ([GitHub](https://github.com/jdomdev))
- Juan Carlos Macías ([GitHub](https://github.com/juancmacias))
