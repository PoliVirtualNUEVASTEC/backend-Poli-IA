from fastapi import FastAPI
from pydantic import BaseModel
import json
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
import re

# Inicializar FastAPI
app = FastAPI()

# Habilitar CORS (Evitar restricciones en las peticiones desde el frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración del cliente (IA Deepseek) de LM Studio
client = OpenAI(base_url="http://localhost:1234/v1", api_key="meta-llama-3.1-8b-instruct-128k")

#  Cargar dataset (Información o BD de la universidad)
def cargar_dataset():
    try:
        with open("datos.json", "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print("⚠️ No se encontró el archivo datos.json, asegúrate de que exista.")
        return []

dataset = cargar_dataset()

# Modelo de datos esperado desde el frontend
class ChatRequest(BaseModel):
    messages: list

# Función para encontrar información relevante en el dataset
# (se usará solo con el último mensaje del usuario para encontrar contexto adicional)
def buscar_informacion(pregunta):
    informacion_relevante = []
    pregunta = pregunta.lower()

    for item in dataset:
        try:
            # Convertir todo el contenido del item a texto plano para búsqueda
            texto_item = json.dumps(item, ensure_ascii=False).lower()

            # Si alguna palabra clave de la pregunta aparece en el texto del item
            if any(palabra in texto_item for palabra in pregunta.split()):
                # Armar respuesta solo con los campos disponibles
                info = []

                for clave in ['area', 'ubicacion', "nombre", 'horarios', 'contacto', 'descripcion', 'tramites',  "preguntas", "pregunta", "respuesta", "enlace"]:
                    if clave in item:
                        info.append(f"{clave.capitalize()}: {item[clave]}")

                informacion_relevante.append("\n".join(info))

        except Exception as e:
            print(f"Error procesando item: {e}")

    return "\n\n".join(informacion_relevante) if informacion_relevante else None

# Función para limpiar respuestas de la IA

def limpiar_respuesta(respuesta):
    respuesta = re.sub(r"<think>.*?</think>", "", respuesta, flags=re.DOTALL)
    respuesta = re.sub(r"\n{2,}", "\n", respuesta).strip()

    # Eliminar delimitadores no deseados
    respuesta = re.sub(r"\\\[|\\\]", "", respuesta)
    respuesta = re.sub(r"\[\\boxed{(.*?)}\]", r"$\boxed{\1}$", respuesta)  # Por si se cuela alguno


    return respuesta

# Ruta para manejar las consultas
@app.post("/chat")
async def chat_with_model(request: ChatRequest):
    mensajes = request.messages
    pregunta_usuario = ""

    # Busca el último mensaje del usuario
    for m in reversed(mensajes):
        if m["role"] == "user":
            pregunta_usuario = m["content"]
            break

    contexto = buscar_informacion(pregunta_usuario)

    # Si encontramos información en el dataset, la agregamos al mensaje del usuario
    if contexto:
        mensajes.append({"role": "system", "content": f"Usa esta información institucional: {contexto}"})

    # Siempre aseguramos que el mensaje system inicial esté presente al comienzo
    mensajes.insert(0, {
        "role": "system",
        "content": (
            "Eres un chatbot llamado Poli-IA creado para responder en español preguntas relacionadas con la Universidad Politécnico Colombiano Jaime Isaza Cadavid. "
            "Debes responder con base en la información proporcionada. Si no tienes datos suficientes, ofrece una respuesta general o indica que no puedes ayudar. "
            "Cuando uses expresiones matemáticas, usa solo sintaxis LaTeX válida y bien delimitada. "
            "Utiliza `$...$` para expresiones en línea y `$$...$$` para expresiones en bloque. "
            "No uses `\\[ \\]`, `\\boxed{}`, corchetes u otras notaciones que no sigan los delimitadores `$` o `$$`. "
            "Ejemplos válidos:\n"
            "- $E = mc^2$\n"
            "- $$\\frac{a + b}{2}$$\n"
            "Evita cualquier símbolo extraño o fuera del estándar LaTeX. "
            "Además, mantén el texto limpio, bien formateado y fácil de leer, sin usar etiquetas ni caracteres codificados. "
        )
    })

    try:
        response = client.chat.completions.create(
            model="meta-llama-3.1-8b-instruct-128k",
            messages=mensajes,
            temperature=0.7
        )

        respuesta_limpia = limpiar_respuesta(response.choices[0].message.content)
        return {"response": respuesta_limpia}
    except Exception as e:
        print("⚠️ Error al conectar con la IA de LM Studio:", str(e))
        return {"response": "⚠️ Error al consultar a la IA"}

# Ejecutar el backend
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
