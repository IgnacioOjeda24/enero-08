import requests
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

def generate_response(prompt):
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
  
    outputs = model.generate(
        **inputs, 
        max_length=100,        
        num_return_sequences=1, 
        temperature=0.3,       
        top_k=50,              
        top_p=0.9,             
        no_repeat_ngram_size=2, 
        do_sample=True         
    )
    
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def chat():
    print("¡Bienvenido! Puedes hacerme preguntas.")
    while True:
        user_input = input("Tú: ")
        if user_input.lower() == "salir":
            print("¡Hasta luego!")
            break
        prompt = f"Pregunta: {user_input}\nRespuesta:"  
        response = generate_response(prompt)
        print(f"Modelo: {response}")

if __name__ == "__main__":
    chat()