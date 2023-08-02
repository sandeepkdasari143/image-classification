import uvicorn
from fastapi import FastAPI, UploadFile, File
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchinfo import summary
from going_modular import data_setup, engine
import os
import zipfile
from pathlib import Path
import requests
from typing import List, Tuple
from PIL import Image
import random
import shutil

app = FastAPI()

def unZipData():
    data_path = Path("data/")
    image_path = data_path / "animal_classification"

    # If the image folder doesn't exist, download it and prepare it...
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

        # Download pizza, steak, sushi data
        with open(data_path / "animal_classification.zip", "wb") as f:
            request = requests.get("https://github.com/karthikbhandary2/Images-classification/raw/main/data/animal_classification.zip")
            print("Downloading data...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / "animal_classification.zip", "r") as zip_ref:
            print("Unzipping pizza, steak, sushi data...")
            zip_ref.extractall(image_path)

        # Remove .zip file
        os.remove(data_path / "animal_classification.zip")
    train_dir = image_path / "data/train"
    test_dir = image_path / "data/test"
    return train_dir, test_dir

def prepocessing(train_dir, test_dir):

    manual_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # 1. Reshape all images to 224x224 (though some models may require different sizes)
    transforms.ToTensor(), # 2. Turn image values to between 0 & 1
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,test_dir=test_dir,transform=manual_transforms,batch_size=32)


    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights
    model = torchvision.models.efficientnet_b0(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Get the length of class_names (one output unit for each class)
    output_shape = len(class_names)

    # Recreate the classifier layer and seed it to the target device
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280,out_features=output_shape,bias=True))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, train_dataloader, test_dataloader, class_names, optimizer, loss_fn


def model_training(model, train_dataloader, test_dataloader, optimizer, loss_fn):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    results = engine.train(model=model,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=7,
                        device="cpu")
    return model


# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred_image(model: torch.nn.Module,
                        image_path: str, 
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device="cpu"):
    # 2. Open image
    img = Image.open(image_path)

    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    ### Predict on image ### 

    # 4. Make sure the model is on the target device
    # model

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0)

        # 7. Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image)

        # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

        # 9. Convert prediction probabilities -> prediction labels
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 10. Plot image with predicted label and probability 
    # plt.figure()
    # plt.imshow(img)
    # plt.axis(False);
        accuracy = target_image_pred_probs.max()
        return class_names[target_image_pred_label], accuracy

# Save the model to an .h5 file
def save_model_to_h5(model, filename):
    model.eval()  # Set model to evaluation mode before saving
    torch.save(model.state_dict(), filename)

# Load the model from an .h5 file
def load_model_from_h5(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()  # Set model to evaluation mode after loading
    return model


@app.post('/classify/')
def classify_image(file: UploadFile = File(...)):

    UPLOAD_DIRECTORY = "uploads/"
    # Create the upload directory if it doesn't exist
    os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
    temp_file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(temp_file_path, "wb") as image_file:
        shutil.copyfileobj(file.file, image_file)

    # Load the model only once during startup
    model_filename = "models/animal_classification_model.h5"
    if not os.path.exists(model_filename):
        train_dir, test_dir = unZipData()
        model, train_dataloader, test_dataloader, class_names, optimizer, loss_fn = prepocessing(train_dir, test_dir)
        model = model_training(model, train_dataloader, test_dataloader, optimizer, loss_fn)
        save_model_to_h5(model, model_filename)
    else:
        model = torchvision.models.efficientnet_b0()
        model = load_model_from_h5(model, model_filename)
    
    # Define transformation for the image
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Predict on custom image
    predicted_class, accuracy = pred_image(model=model,
                        image_path=temp_file_path,
                        class_names=class_names,
                        transform=image_transform)

    #Delete Temperory Image
    os.remove(temp_file_path)

    if accuracy < 0.5:
        return  {
        "status": 200,
        "message": "safe",
        "predicted_class": predicted_class,
        "accuracy": accuracy
    }
    else:
        return {
            "status": 400,
            "message": "unsafe",
            "predicted_class": predicted_class,
            "accuracy": accuracy
        }

@app.get('/')
def index():
    return {
        "message": "Please upload any image"
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)


#uvicorn main:app --reload4