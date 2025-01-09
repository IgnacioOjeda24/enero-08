#Imported Python libraries.
import os #To create the REST API.
from fastapi import FastAPI #Hugging Face library for using pre-trained models.
from pydantic import BaseModel #Used for data validation in FastAPI.
from transformers import pipeline #ASGI server to run the API.


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = FastAPI()


text_generation_model = pipeline("text-generation", model="gpt2")
text_summarization_model = pipeline("summarization", model="facebook/bart-large-cnn")

class ModelSwitcher:
    """
    Clase que permite generar texto con LLM y luego resumirlo con summarization.
    """
    def __init__(self):
        self.models = {
            "LLM": text_generation_model,
            "summarization": text_summarization_model
        }

    def generate_and_summarize(self, text: str):
        """
        Genera texto con el modelo LLM y luego lo resume usando el modelo de resumen.
        """

        generated_text = self.models["LLM"](text, max_length=100, num_return_sequences=1)[0]["generated_text"]

        summary = self.models["summarization"](generated_text, max_length=60, min_length=30, do_sample=False)[0]["summary_text"]
        
        return {"generated_text": generated_text, "summary": summary}


model_switcher = ModelSwitcher()

class ModelRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de generación y resumen de texto."}

@app.post("/generate_and_summarize/")
def generate_and_summarize(request: ModelRequest):
    """
    Recibe un texto, lo pasa por el modelo LLM y luego lo resume.
    """
    result = model_switcher.generate_and_summarize(request.text)
    return result