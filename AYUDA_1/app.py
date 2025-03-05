import easyocr
import os
import cv2
import json
from google.colab import files
from IPython.display import display, Image
from transformers import pipeline

def extract_text_from_image(image_path):
    # Verificar si la imagen existe
    if not os.path.exists(image_path):
        print(f"No se encontró la imagen '{image_path}'. Por favor, súbela.")
        uploaded = files.upload()  # Pide al usuario que suba una imagen
        if not uploaded:
            raise ValueError("No se subió ninguna imagen.")
        image_path = list(uploaded.keys())[0]  # Usa la imagen subida

    # Mostrar la imagen
    print("🖼️ Imagen cargada:")
    display(Image(filename=image_path))

    # Crear el lector de EasyOCR (en español)
    reader = easyocr.Reader(['es'])

    # Leer el texto de la imagen
    result = reader.readtext(image_path, detail=0)

    # Concatenar el texto extraído
    extracted_text = "\n".join(result)
    return extracted_text

resultado = extract_text_from_image("test 1.png")
print(resultado)

# Crear un pipeline de extracción de información
extractor = pipeline("question-answering", model="mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es")

preguntas = [
    "¿Cuál es el nombre completo del remitente?",
    "¿Cuál es el número de cuenta bancaria del remitente?",
    "¿Cuál es el banco del remitente?",
    "¿Cuál es la CLABE interbancaria del remitente?",
    "¿Cuál es la dirección del remitente?",
    "¿Cuál es la fecha de la transferencia?",
    "¿Cuál es la hora de la transferencia?",
    "¿Cuál es el monto a transferir?",
    "¿Cuál es el concepto de la transferencia?",
    "¿Cuál es la referencia de la transferencia?",
]

# Extraer respuestas
datos_extraidos = {}
for pregunta in preguntas:
    respuesta = extractor(question=pregunta, context=resultado)
    clave = pregunta.lower().replace("¿cuál es ", "").replace("?", "").replace(" ", "_")
    datos_extraidos[clave] = respuesta["answer"]

# Convertir a JSON
json_datos = json.dumps(datos_extraidos, indent=4, ensure_ascii=False)

# Imprimir resultados
print(json_datos)