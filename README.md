# Flower-Identification-using-Image-classification
This is a CNN model using PyTorch libraries to identify Dandelions and Daisies.Image classification is used to identify the correct catergory of the image that it falls in to (Daisy or Dandelion)

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim

from google.colab import drive
drive.mount('/content/drive')

# Defining data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets 
train_data = ImageFolder('file_path', transform=transform)
val_data = ImageFolder('file_path', transform=transform)
test_data = ImageFolder('file_path', transform=transform)

# Data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


#CNN architecture
class FlowerClassifier(nn.Module):
    def __init__(self):
        super(FlowerClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes: daisy and dandelion

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training loop
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader)}")

        # Validation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            val_accuracy = correct / total
            print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

  #Model Training
train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10)

# Evaluating the model on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

#Testing the Model for the first 5 Images in the dataset
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_data = ImageFolder('file_path', transform=transform)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

class_names = test_data.classes
model.eval()  

# Function to predict and display images
def predict_images(model, test_loader, num_images=5):
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            if i >= num_images:
                break
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            image = inputs.squeeze(0).permute(1, 2, 0).cpu().numpy()
            plt.imshow(image)
            plt.title(f'Predicted: {class_names[predicted.item()]}, Actual: {class_names[labels.item()]}')
            plt.axis('off')
            plt.show()

# Testing the model on the first 5 images
predict_images(model, test_loader, num_images=5)

#Outut:

![image](https://github.com/Pavanirt/Flower-Identification-using-Image-classification/assets/160448544/6a563912-7bf9-4f8b-9c74-8a7af8cc2995)

![image](https://github.com/Pavanirt/Flower-Identification-using-Image-classification/assets/160448544/3656cfaf-26c0-4262-9209-1702c7c6a037)

![image](https://github.com/Pavanirt/Flower-Identification-using-Image-classification/assets/160448544/e9585bc6-6706-41e1-95fa-a9f72857f3d0)

![image](https://github.com/Pavanirt/Flower-Identification-using-Image-classification/assets/160448544/26de892f-11f8-4966-9406-087cc2a6f016)

![image](https://github.com/Pavanirt/Flower-Identification-using-Image-classification/assets/160448544/4f8cccc2-ec7b-4759-af25-1a7ffd6ea173)


