from .DataSet.py import DataSetLoader
from .ModelLoader.py import ModelLoader
import os
import sys
from torch import nn, optim

# The pretrained network was got from
# https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/torchvision_models.py

# Setting up
model_folder = path.append(sys.getcwd(), 'pretrained_model')
test_image_names = ['12M Too Close.jpg', 'Full HD (2M) Too Close.jpg', 'HD (092M) Too Close.jpg', 'SVGA (048M) Too Close.jpg']
test_root_folder = os.path.join(sys.getcwd(), '..\\..\\DataSet\\Single face - different range and quality\\Too Close')
labels = ['Really big file', 'Big file', 'Small file', 'Realy small file']

test_dataset_loader = DataSetLoader(test_root_folder, test_image_names, labels) # Поправить пути
train_dataset_loader = DataSetLoader(root_folder, image_names, labels) # Поправить пути
model_loader = ModelLoader('https://download.pytorch.org/models/resnet50-19c8e357.pth', model_folder)

criterion = nn.NLLLoss()
optimizer = optim.Adam(neural_network_model.fc.parameters(), lr=0.003)

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
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device),
                                      labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = 
                        top_class == labels.view(*top_class.shape)
                    accuracy +=
                   torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
torch.save(model, 'aerialmodel.pth')
