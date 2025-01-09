#Libraries required for this project.
import torch #To perform operations with tensors and run the PyTorch model.
from transformers import ViTForImageClassification, ViTImageProcessor 
import matplotlib.pyplot as plt #To graphically display the image and prediction results of the first 7 objects.
from PIL import Image #To upload and process images.
import easygui #To display a dialog box to select images interactively.


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Determines whether the code will run on GPU (cuda) or CPU, depending on availability.


model_name = "google/vit-base-patch16-224" #Defines the name of the pretrained Vision Transformer (ViT) model that will be loaded into this first project.
model = ViTForImageClassification.from_pretrained(model_name).to(device) #Loads the ViT model for image classification and moves it to the device (GPU or CPU).
processor = ViTImageProcessor.from_pretrained(model_name) #Loads the processor corresponding to the ViT model, which is used to prepare the images.

model.eval() #Set the model to evaluation mode, disabling gradient calculation to optimize resources.

def predict_with_transformers(image_path, topk=7):
    """
    Realiza una predicción utilizando un modelo Vision Transformer.
    
    Args:
        image_path (str): Ruta a la imagen.
        topk (int, opcional): Número de clases más probables a mostrar. Lo defino por 7.
        
    Returns:
        list: Lista de tuplas (clase, probabilidad) para las clases más probables.
    """

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

        top_probs, top_indices = torch.topk(probs, topk)
    

    top_probs = top_probs.squeeze().tolist()
    top_indices = top_indices.squeeze().tolist()
    labels = model.config.id2label
    predicted_classes = [labels[idx] for idx in top_indices]
    
    return list(zip(predicted_classes, top_probs))


while True:

    image_path = easygui.fileopenbox(title="Select an image:", filetypes=["*.jpg", "*.png", "*.jpeg", "*.bmp"])

    if not image_path: 
        print("The image was not selected successfully, so the process will end.")
        break


    predictions = predict_with_transformers(image_path)
    print("\nPrediction results:")
    for class_name, prob in predictions:
        print(f"{class_name}: {prob:.4%}")


    plt.imshow(Image.open(image_path))
    plt.title(f"Main prediction: {predictions[0][0]}")
    plt.axis("off")
    plt.show()

    another_image = input("Do you want to upload another image? (y/n): ").strip().lower()
    if another_image != 'y':
        print("Process finished.")
        break

