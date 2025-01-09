# Proyecto_parte_1: 
Instrucción:

Proyecto 1 : Utilizando un api de hugginface de un modelo LLM crea una aplicación donde el usuario pueda realizar preguntas y el modelo le responda, en el readme especifica el modelo usado y los parámetros en la petición junto con un ejemplo de uso.

# Selección del modelo de Hugging Face: 

Este proyecto utiliza un modelo Vision Transformer (ViT) para clasificar las imágenes y tener una predicción con porcentaje de los primeros 7 objetos con más probabilidad (4 decimales), se ejecuta de manera local utilizando Pytorch y los modelos preentrenados disponibles en Hugging Face.

## Requisitos para la Ejecución

### Dependencias

1. **Python 3.8 a 3.10 independiente del sistema operativo que tenga el usuario como: Windows, Linux o Mac.**
2. **Visual studio instalado independiente del sistema operativo que tenga el usuario como: Windows, Linux o Mac.**
3. **Instalar el siguiente comando:**

   ``` bash
   pip install torch torchvision torchaudio transformers matplotlib pillow easygui
   
4. **Librerías necesarias de Python instaladas**:
   - `torch`
   - `transformers`
   - `matplotlib`
   - `easygui`
5. **Librerías instaladas en el proyecto que se adjunta una imagen al inicio del archivo script_modelo.py**.

![U1](https://github.com/user-attachments/assets/10fde574-a813-4a83-bc0f-826db50c0252)
   
6. **Antes de correr el programa, instalar los siguientes comandos:**

   Para crear el entorno virtual en visual studio code.

   ``` bash
   python -m venv nombre_propio_de_entorno virtual

   Ejemplo:

   ``` bash
   python -m venv proyecto_2

   Para activar el entorno virtual en visual studio code.

   ``` bash
   \nombre_propio_de_entorno virtual\Scripts\activate

   Ejemplo:

   ``` bash
   \proyecto_2\Scripts\activate

7. **Hardware: GPU (no obligatorio): Si usted tiene una CPU compatible, la librería PyTorch la utilizará de forma automática para acelerar la ejecución, en el caso de lo contrario el modelo funcionará exitosamente en la CPU, aunque será menos eficiente. Si no tienes una, el modelo funcionará correctamente en la CPU, aunque será más lento.**


## Para que funcione el proyecto correctamente independiente del dispositivo.

1. **Clonar el proyecto con el siguiente comando en visual studio code:**
2. **Verifica si está todo instalado, sino falta uno de ellos hay que revisar con los comandos que salieron anteriormente como las librerías, crear el entorno, activación del entorno, entre otros**.
3. **Si el usuario desea andar el proyecto tiene que ejecutar el siguiente comando:**
   ```bash
   python .\script_modelo.py
4. **A medida que se cargue el proyecto, al usuario le va a aparecer un cuadro de diálogo donde adjuntará la foto de manera local o lo que tiene el PC, se importará correctamente, siempre y cuando sea el formato jpg, png, bmp y jpeg**.

   ![U6](https://github.com/user-attachments/assets/e9530d90-176a-4570-a519-fcb0c450b0ce)

5. **Después que se haya importado correctamente la imagen**

   **En el caso de no adjuntar una imagen**

   ![U2](https://github.com/user-attachments/assets/5fe81ae7-9cca-4b0f-b4bc-7967511639a0)

6. **Luego, aparecerá en una interfaz gráfica la imagen que se adjuntó y su predicción principal (Osea el que tiene más porcentaje).**

   ![U3](https://github.com/user-attachments/assets/2685c93d-1293-4a2e-9990-219d8b3008c9)

   
7. **Luego, realizará la predicción y mostrará las primeras 7 clases probables y sus probabilidades con 4 decimales en la terminal de power shell de visual studio code**.

   Ejemplo:

   ![U4](https://github.com/user-attachments/assets/014065b7-c318-47b1-9813-7fc6dce7e3c0)


8. **Para seguir el proceso o terminar el proceso, tiene que presionar el botón x de la ventana**.

    ![u6](https://github.com/user-attachments/assets/c3a341ac-b572-4210-a6c8-9a3f13b913d8)

   
10. **Si el usuario desea con el proceso poner una y en el caso no continua con n**.
    **En el caso de sí.**
   
    ![U7](https://github.com/user-attachments/assets/a5a1f379-b11e-41bf-a518-8e93c064fc3a)


    **En el caso de no.**

    ![U8](https://github.com/user-attachments/assets/6b4fc75c-aec4-4b95-86d8-bb0f938077b4)


11. **Así sucesivamente sigue la consulta de la imagen, esto es un ciclo, hasta que usuario determine ejecutar n en el momento que desee.**

# Sección sobre Transformers y Ejecución de Modelos en Local: 

## ¿Qué son los Transformers?: 

Los transformers es un marco de código abierto para el aprendizaje profundo creado por la empresa estadounidense Hugging face en año 2016, por lo que proporciona API y herramientas para descargar modelos previamente entrenados de última generación y optimizarlos para tener el mayor rendimiento posible. Existen modelos como: NLP (procesamiento de lenguaje natural), clasificación de vídeos, clasificación de textos, clasificación de tokens, entre otros.

El modelo que se trabajará en esta ocasión es de Vision Transformer (ViT) que son versiones adaptadas para trabajar con imágenes en python, JS, entre otros lenguajes de programación.

Un modelo transformer se basa en el mecanismo de autoatención, que permite al modelo aprender a centrarse en diversas partes de la entrada (en este caso, la imagen) mientras procesa la información. Esto lo convierte en una arquitectura muy eficaz para tareas como la clasificación de imágenes y decir la predicción de los objetos en la imagen.

## Ejecución de Modelos en Local:

Este proyecto utiliza el modelo google/vit-base-patch16-224, un modelo Vision Transformer preentrenado disponible en la empresa mencionada anteriormente. La ejecución de modelos transformers en local se hace posible gracias a bibliotecas como transformers y torch, que permiten cargar modelos preentrenados y realizar inferencias de manera eficiente tanto en CPU como en GPU.

## Lo que Aprendí:

En este proyecto, aprendí a trabajar con modelos preentrenados de transformers para tareas de visión por computadora, como la clasificación de imágenes. El proceso de usar un modelo transformer localmente es bastante directo y eficiente, especialmente cuando se trabaja con GPUs, ya que estos modelos son computacionalmente intensivos.

Además, comprendí mejor cómo el procesamiento de imágenes con modelos transformers es similar a otros tipos de entrada, como texto, pero con ciertos cambios, como el uso de patching para dividir las imágenes en pequeñas secciones que el modelo puede procesar de manera más objetiva.


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
