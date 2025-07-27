from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
import json
from sklearn.model_selection import train_test_split


# 1. Cargar el Dataset
# Usaremos el dataset IMDb para análisis de sentimiento de películas.
# Contiene reseñas de películas etiquetadas como 'positive' (1) o 'negative' (0).
try:
    print("Cargando el dataset IMDb...")
    dataset = load_dataset("imdb")
    print("Dataset cargado. Ejemplos:")
    print(dataset["train"][0])
    print(dataset["test"][0])
except Exception as e:
    print(f"Error al cargar el dataset: {e}")
    exit(1)

print("\nDividiendo el dataset: 60% entrenamiento, 40% validación...")
train_data = dataset["train"]
test_data = dataset["test"]
indices = list(range(len(train_data)))
train_idx, val_idx = train_test_split(indices, test_size=0.4, random_state=42)
train_split = train_data.select(train_idx)
val_split = train_data.select(val_idx)
dataset["train"] = train_split
dataset["validation"] = val_split
dataset["test"] = test_data


# 2. Cargar el Tokenizador
# Usaremos un tokenizador de un modelo pre-entrenado pequeño (DistilBERT)
# que es eficiente y bueno para fine-tuning.
try:
    print("\nCargando el tokenizador DistilBERT...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
except Exception as e:
    print(f"Error al cargar el tokenizador: {e}")
    exit(1)


# 3. Función de Preprocesamiento
# Esta función tokenizará el texto y lo preparará para el modelo.
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

print("\nPreprocesando el dataset...")
try:
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
except Exception as e:
    print(f"Error al preprocesar el dataset: {e}")
    exit(1)


# 4. Cargar el Modelo
# Cargamos un modelo pre-entrenado para clasificación de secuencias.
# Especificamos el número de etiquetas (2: positivo/negativo).
try:
    print("\nCargando el modelo DistilBERT para clasificación de secuencias...")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit(1)


# 5. Definir Métricas de Evaluación
# Usaremos Accuracy, F1-score, Precision y Recall.
print("\nDefiniendo métricas de evaluación...")
metric = load("accuracy")
f1_metric = load("f1")
precision_metric = load("precision")
recall_metric = load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")
    recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted")
    return {**accuracy, **f1, **precision, **recall}


# 6. Configurar Argumentos de Entrenamiento
# Aquí definimos cómo se entrenará el modelo (épocas, tamaño de batch, etc.).
print("\nConfigurando argumentos de entrenamiento...")
training_args = TrainingArguments(
    output_dir="./results",          # Directorio para guardar los resultados
    num_train_epochs=3,              # Número de épocas de entrenamiento
    per_device_train_batch_size=16,  # Tamaño del batch por dispositivo (GPU/CPU)
    per_device_eval_batch_size=16,   # Tamaño del batch para evaluación
    warmup_steps=500,                # Número de pasos para el calentamiento del learning rate
    weight_decay=0.01,               # Regularización L2
    logging_dir="./logs",            # Directorio para los logs de TensorBoard
    logging_steps=100,
    report_to="none"                # No reportar a ninguna plataforma (ej. wandb)
)


# 7. Crear el Trainer
# El Trainer es una clase de Hugging Face que simplifica el entrenamiento.
print("\nCreando el Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# 8. Entrenar el Modelo
if torch.cuda.is_available():
    print(f"\nGPU detectada: {torch.cuda.get_device_name(0)}")
    print("Entrenamiento se realizará en GPU.")
else:
    print("\nNo se detectó GPU. El entrenamiento se realizará en CPU y será más lento.")

try:
    trainer.train()
    print("\nEntrenamiento completado.")
except Exception as e:
    print(f"Error durante el entrenamiento: {e}")
    exit(1)


# 9. Evaluar el Modelo Final
print("\nEvaluando el modelo final en el conjunto de prueba...")
try:
    eval_results = trainer.evaluate(tokenized_dataset["test"])
    print(f"Resultados de la evaluación final: {eval_results}")
    # Guardar métricas en JSON
    with open("./results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4, ensure_ascii=False)
    print("Métricas guardadas en ./results/metrics.json")
except Exception as e:
    print(f"Error en la evaluación: {e}")


# 10. Guardar el Modelo
# Guardamos el modelo entrenado y el tokenizador para futuras inferencias.
model_save_path = "./fine_tuned_sentiment_model"
try:
    print(f"\nGuardando el mejor modelo y tokenizador en: {model_save_path}")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print("Modelo y tokenizador guardados exitosamente.")
except Exception as e:
    print(f"Error al guardar el modelo: {e}")

