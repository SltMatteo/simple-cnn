import torch 
import numpy as np
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
import torch.utils.data
import torchvision 
import torchvision.transforms as transforms
import torch.utils
from PIL import Image
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module): 
    def __init__(self): 
        r"""

        ARCHITECTURE :

        input: 3 x 32 x 32 (RGB image of 32x32 px)
              ↓
        ---------------
        |             |
        |   Conv1     | -> 32 filters of size 3x3, output shape: 32 x 32 x 32
        |             |
        ---------------
            ↓ (ReLU)
        ---------------
        |             |
        |  MaxPool1   | -> 2x2 kernel, output shape: 32 x 16 x 16 (spatial dimensions halved)
        |             |
        ---------------
            ↓
         ---------------
        |             |
        |   Conv2     | -> 64 filters of size 3x3, output shape: 64 x 16 x 16
        |             |
        ---------------
            ↓ (ReLU)
        ---------------
        |             |
        |  MaxPool1   | -> 2x2 kernel, output shape: 64 x 8 x 8 (spatial dimensions halved)
        |             |
        ---------------
            ↓ 
        ---------------
        |   Flatten   | -> output shape: 4096 (64 * 8 * 8 = 4096)
        ---------------
            ↓ 
        ---------------
        |             |
        |     FC1     | -> output shape: 128
        |             |
        ---------------
            ↓ (ReLU)
        ---------------
        |             |
        |     FC2     | -> output shape: 10 (one for each class in CIFAR-10)
        |             |
        ---------------
            ↓ 
            output : 10 class scores (one for each CIFAR-10 class)


    """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) 
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128) 
        self.fc2 = nn.Linear(128, 10) 

    def forward(self, x): 
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 *8) #flattening 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        return x 



def load_data(batch_size): 
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), #flips the image with probability 1/2 
        transforms.ToTensor(), #np array to pytorch tensor 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #normalization will make the model converge faster during trainer 
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True) #wraps the train dataset into a loader 

    #same for test set 
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, num_epochs): 
    model.train() #put the model in train mode 

    for epoch in range(num_epochs): 
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0): 
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs) 
            loss = criterion(outputs, labels) 

            loss.backward()
            optimizer.step()

            running_loss+=loss.item() 
            if i%10 == 9: 
                print(f'[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print("Finished training - savind model") 
    torch.save(model.state_dict(), 'cnn_model.pth') 


def evaluate_model(model, test_loader):

    model.load_state_dict(torch.load('cnn_model.pth'))
    model.eval() #put the model in eval mode 
    correct = 0
    total = 0 
    with torch.no_grad(): 
        for data in test_loader: 
            images, labels = data 
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) 
            _, predicted = torch.max(outputs.data, 1)
            total+= labels.size(0) 
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')

def single_prediction(model, image_path): 

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    #have the image have the correct format 
    transform = transforms.Compose([
        transforms.Resize((32, 32)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = Image.open(image_path).convert('RGB')

    image = transform(image).unsqueeze(0) #add batch dimension (1, 3, 32, 32) 

    image = image.to(device) 

    model.eval() 

    with torch.no_grad(): 
        output = model(image) 
        probabilities = F.softmax(output, dim=1)
        _, predicted = torch.max(probabilities, 1)
        predicted_class = predicted.item()


    for i, prob in enumerate(probabilities[0]):
        print(f"Class {i} ({classes[i]}): {prob.item()*100:.2f}%")

    # Print the top prediction
    print(f'\nPredicted Class: {predicted_class} ({classes[predicted_class]})')

import random 


def choose_random_image(folder_path):
    images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    
    if not images:
        print("No .jpg images found in the specified folder.")
        return None
    
    random_image = random.choice(images)
    image_path = os.path.join(folder_path, random_image)

    print('\nChosen image: ', image_path)
    print('\n')
    
    return os.path.join(folder_path, random_image)



def main():
    model_path = 'cnn_model.pth'
    folder_path = 'images/mixed'
    image_path = choose_random_image(folder_path)

    if os.path.exists(model_path):
        print("Model exists, loading model and making a prediction...")
        model = CNN().to(device)
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
        print('\n')
        single_prediction(model, image_path)

    else:
        print("Model does not exist. Training a new model...")

        #hyperparameters
        batch_size = 64
        lr = 0.0001
        num_epochs = 10

        train_loader, test_loader = load_data(batch_size)

        model = CNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_model(model, train_loader, criterion, optimizer, num_epochs)

        torch.save(model.state_dict(), model_path)
        print(f"Model saved as '{model_path}'")

        single_prediction(model, image_path)

if __name__ == "__main__":
    main()

