# proyecto_2.2

Instrucción:
Huggingface Api
Proyecto 1 : Utilizando un api de hugginface de un modelo LLM crea una aplicación donde el usuario pueda realizar preguntas y el modelo le responda, en el readme especifica el modelo usado y los parámetros en la petición junto con un ejemplo de uso.

## Resumen del proyecto parte 2.

Este proyecto implementa una aplicación de chatbot, que el usuario haga cualquier pregunta con el modelo LLM que se llama openai-community/gpt2, responde de manera no exacta, pero funciona el código. Lo ideal que se haga preguntas en inglés, ya que tiene respuestas con más precisión respecto a los demás idiomas.

## Requisitos para la Ejecución.

1. **Python 3.8 a 3.10 independiente del sistema operativo que tenga el usuario como: Windows, Linux o Mac**.
2. **Visual studio instalado independiente del sistema operativo que tenga el usuario como: Windows, Linux o Mac.**
3. **Instalar el siguiente comando**
   
   ```bash
   pip install transformers torch requests

4. **Librerías instaladas en python**

   - `transformers: para interactuar con los modelos de Hugging Face`
   - `torch: para la ejecución de los modelos.`
   - `requests: para enviar peticiones HTTP si se decide utilizar la API de Hugging Face`

5. **Librerías importadas en foto**

   ![U20](https://github.com/user-attachments/assets/1de1414b-b5be-4c7b-89e6-8ce169a6f6e2)

6. **Antes de correr el programa, instalar los siguientes comandos:**

   Para crear el entorno virtual en visual studio code.

   ``` bash
   python -m venv nombre_propio_de_entorno virtual

   Ejemplo:

   ``` bash
   python -m venv env2

   Para activar el entorno virtual en visual studio code.

   ``` bash
   \nombre_propio_de_entorno virtual\Scripts\activate

   Ejemplo:

   ``` bash
   \env2\Scripts\activate


## Requisitos para la Ejecución.

1. **Clonar el proyecto con el siguiente comando independiente del sistema operativo**.
2. **Verifica si está todo instalado, sino falta uno de ellos hay que revisar con los comandos que salieron anteriormente como las librerías, crear el entorno, activación del entorno, entre otros**.
3. **Si el usuario desea andar el proyecto tiene que ejecutar el siguiente comando:**

   ``` bash
   python app.py

4. **Programa funcionando, pero no responderá correctamente las preguntas, ya que este chat-gpt-2, no tiene la exactitud de responder las preguntas.**

   ![U23](https://github.com/user-attachments/assets/3dd40ddc-e661-4821-b327-78eac8bccb35)

   
5. **Salir de la aplicación: Si deseas salir de la aplicación, simplemente escribe salir y presiona Enter.**


#### Parámetros utilizados en la generación:

El método `model.generate` controla cómo el modelo produce texto basado en una entrada. A continuación, se describen los parámetros utilizados:

- **`max_length`**: Longitud máxima de la respuesta generada, medida en tokens (palabras o fragmentos de palabras).
- **`num_return_sequences`**: Número de respuestas generadas para cada entrada. Aquí se genera únicamente una.
- **`temperature`**: Ajusta la aleatoriedad de la salida. 
- **`top_k`**: Limita las palabras consideradas por el modelo a las 50 más probables en cada paso.
- **`top_p`**: Activa el "nucleus sampling", que toma en cuenta solo las palabras más probables cuya suma de probabilidades sea ≤ 0.9.
- **`no_repeat_ngram_size`**: Impide la repetición de patrones consecutivos con 2 o más palabras.
- **`do_sample`**: Activa el muestreo aleatorio, permitiendo respuestas más variadas.
- **NOTA: Lo hice por defecto para no complicar el programa y no decía específicamente lo que se pedía.**

#### Ejemplo.

![U24](https://github.com/user-attachments/assets/4377140a-b79f-42c8-89b0-f3c0f01e0b78)



