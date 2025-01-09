# app.py

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()


text_generation_model = pipeline("text-generation", model="gpt2")
text_summarization_model = pipeline("summarization", model="facebook/bart-large-cnn")

class ModelSwitcher:
    """
    Clase que permite iterar entre modelos de tipo LLM y modelos de resumen.
    """
    def __init__(self):
        self.models = {
            "LLM": text_generation_model,
            "summarization": text_summarization_model
        }

    def switch_model(self, model_type: str, text: str):
        """
        Cambia de modelo y ejecuta el modelo solicitado (LLM o resumen).
        """
        if model_type not in self.models:
            return {"error": "Modelo no válido. Use 'LLM' o 'summarization'."}
        
        model = self.models[model_type]
        
        if model_type == "LLM":

            result = model(text, max_length=100, num_return_sequences=1)
            return {"generated_text": result[0]["generated_text"]}
        
        elif model_type == "summarization":

            result = model(text, max_length=60, min_length=30, do_sample=False)
            return {"summary": result[0]["summary_text"]}


model_switcher = ModelSwitcher()

class ModelRequest(BaseModel):
    model_type: str
    text: str

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de cambio de modelos (LLM <-> Resumen)"}

@app.post("/switch_model/") 
def switch_model(request: ModelRequest):
    """
    Recibe un tipo de modelo (LLM o summarization) y el texto, y devuelve el resultado.
    """
    result = model_switcher.switch_model(request.model_type, request.text)
    return result
