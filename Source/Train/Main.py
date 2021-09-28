from .DataSet import DataSetLoader
from .ModelLoader import ModelLoader
import os
import sys
from torch import nn, optim
import torch

# The pretrained network was got from
# https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/torchvision_models.py
# https://github.com/pytorch/vision/tree/master/torchvision/models

# Setting up
model_file_name = 'roman_resnet50.pth'
model_folder = os.path.append(sys.getcwd(), 'pretrained_model')
test_image_names = ['12M Too Close.jpg', 'Full HD (2M) Too Close.jpg', 'HD (092M) Too Close.jpg', 'SVGA (048M) Too Close.jpg']
test_root_folder = os.path.join(sys.getcwd(), '..\\..\\DataSet\\Single face - different range and quality\\Too Close')
labels = ['Really big file', 'Big file', 'Small file', 'Realy small file']

test_dataset_loader = DataSetLoader(test_root_folder, test_image_names, labels) # Поправить пути
train_dataset_loader = DataSetLoader(root_folder, image_names, labels) # Поправить пути
model_loader = ModelLoader('https://download.pytorch.org/models/resnet50-19c8e357.pth', model_folder, model_file_name)
neural_network_model = model_loader.load()
device = model_loader.load_device()

criterion = nn.NLLLoss()
optimizer = optim.Adam(neural_network_model.fc.parameters(), lr = 0.003)

epochs = 10
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []
for epoch in range(epochs):
    for inputs, labels in train_dataset_loader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = neural_network_model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            neural_network_model.eval()
            with torch.no_grad():
                for inputs, labels in test_dataset_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = neural_network_model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(test_dataset_loader))
            test_losses.append(test_loss/len(test_dataset_loader))                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(test_dataset_loader):.3f}.. "
                  f"Test accuracy: {accuracy/len(test_dataset_loader):.3f}")
            running_loss = 0
            neural_network_model.train()

torch.save(neural_network_model, model_file_name)