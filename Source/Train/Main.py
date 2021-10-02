import DataSetModule
import ModelLoaderModule
from torch import nn, optim
import torch
import Utils

# The pretrained network was got from
# https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/torchvision_models.py
# https://github.com/pytorch/vision/tree/master/torchvision/models
# https://download.pytorch.org/models/resnet50-19c8e357.pth

# Setting up
train_image_names, test_image_names = Utils.split_dataset(Utils.get_dataset())
train_dataset_loader = DataSetModule.DataSetLoader(train_image_names)
test_dataset_loader = DataSetModule.DataSetLoader(test_image_names)
model_loader = ModelLoaderModule.ModelLoader()
neural_network_model = model_loader.load()
device = Utils.load_device()

criterion = nn.NLLLoss()
optimizer = optim.Adam(neural_network_model.fc.parameters(), lr = 0.003)

# Start train
epochs = 20
steps = 0
running_loss = 0
print_every = 2
train_losses, test_losses = [], []
for epoch in range(epochs):
    for trdl_inputs, trdl_labels in train_dataset_loader.load():
        steps += 1
        inputs, labels = trdl_inputs.to(device), trdl_labels.to(device)
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
                for tdl_inputs, tdl_labels in test_dataset_loader.load():
                    inputs, labels = tdl_inputs.to(device), tdl_labels.to(device)
                    logps = neural_network_model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim = 1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(test_dataset_loader))
            test_losses.append(test_loss/len(test_dataset_loader))                    
            print(f"Epoch {epoch + 1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(test_dataset_loader):.3f}.. "
                  f"Test accuracy: {accuracy/len(test_dataset_loader):.3f}")
            running_loss = 0
            neural_network_model.train()

torch.save(neural_network_model, Utils.path_to_output_nn())
