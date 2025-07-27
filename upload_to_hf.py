import os
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Ruta local donde guardaste el modelo y tokenizador
folder_path = "./fine_tuned_sentiment_model"
# ID de tu repositorio en Hugging Face (ajusta con tu usuario y nombre deseado)
repo_id = "juancmamacias/jd-jcms"

# Asegúrate de tener el token de acceso en la variable de entorno HF_TOKEN

# Crear el repo si no existe

token = os.getenv("HF_TOKEN")
if not token or not token.startswith("hf_"):
    raise ValueError("El token HF_TOKEN no está definido o no es válido. Revisa tu archivo .env.")

api = HfApi(token=token)

# Validar el token
try:
    user = api.whoami()
    print(f"Token válido. Usuario autenticado: {user['name']}")
except HfHubHTTPError as e:
    raise ValueError(f"Token HF_TOKEN inválido o sin permisos suficientes: {e}")
try:
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    print(f"Repositorio '{repo_id}' creado o ya existente.")
except Exception as e:
    print(f"Advertencia al crear el repositorio: {e}")

api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="model",
)

print("¡Modelo subido correctamente a Hugging Face!")
