from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
from flask_cors import CORS
import collections
collections.Iterable = collections.abc.Iterable
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
app = Flask(__name__)
CORS(app)

class TumorClassifier(nn.Module):
    def __init__(self, num_classes):
        super(TumorClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = TumorClassifier(num_classes=4)
model.load_state_dict(torch.load("./Model/best_model.pth",map_location ='cpu'))
model.eval()

class_name = {0:"glioma_tumor",1:"meningioma_tumor",2:"no_tumor", 3:"pituitary_tumor"}

def loadImg(imgPath): 
    img = tf.keras.preprocessing.image.load_img(imgPath, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array
    

@app.route('/', methods=['POST'])
def home(): 
    transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    
    if request.method == 'POST':  
        print("Received a POST request")
        file = request.files['file']
        file_path = os.path.join("./", file.filename)
        file.save(file_path)
        image = Image.open(file_path)
        print("Image opened:", image)
        
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)  
        # print("Input batch shape: ", input_batch.shape)
        with torch.no_grad(): 
            output = model(input_batch)
        _, predicted_class = torch.max(output, 1)
        print("Predicted class: ",predicted_class)
        predicted_class_idx = predicted_class.item()
        print(class_name[predicted_class_idx])
        
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            print(f"File {file_path} does not exist, skipping deletion.")
        
        return jsonify({"message": "Prediction successful", "predicted_class": class_name[predicted_class_idx]})

if __name__ == '__main__':
    app.run(debug=True)