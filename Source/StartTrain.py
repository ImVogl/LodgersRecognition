import Train.DataSetModule as dsm
import Common.ModelLoaderModule as mlm
import torch
import Common.Utils as utilites
import Train.TrainService as ts

# The pretrained network was got from
# https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/torchvision_models.py
# https://github.com/pytorch/vision/tree/master/torchvision/models
# https://download.pytorch.org/models/resnet50-19c8e357.pth

# Setting up
model_loader = mlm.PretrainedModelLoader()

# Pretrain model.
def Pretrain():
    epochs = 6
    steps = 0
    print_every = 1
    utils = utilites.Utils('..\\..\\DataSet\\VGGDataSet\\FirstEpoche')
    train_image_names, test_image_names = utils.split_dataset(utils.get_dataset(), 0.35)
    train_dataset_loader = dsm.DataSetLoader(train_image_names)
    test_dataset_loader = dsm.DataSetLoader(test_image_names)
    train_losses, test_losses = [], []
    neural_network_model = model_loader.load()
    training_service = ts.TrainService(neural_network_model, 0.0005)
    for epoch in range(epochs):
        train_loss, steps = training_service.train(train_dataset_loader, steps)
        test_loss, accuracy = training_service.eval(test_dataset_loader)
        if steps % print_every == 0:
            message = f"Epoch {epoch + 1}/{epochs} "
            message += f"Train loss: {train_loss/print_every:.3f} "
            message += f"Test loss: {test_loss/len(test_dataset_loader):.3f} "
            message += f"Test accuracy: {accuracy/len(test_dataset_loader):.3f}"
            print(message)
                
        train_losses.append(train_loss/len(test_dataset_loader))
        test_losses.append(test_loss/len(test_dataset_loader))
    
    model_loader.save(neural_network_model)

# Train model.
def Train():
    epochs = 6
    steps = 0
    print_every = 1
    utils = utilites.Utils('..\\..\\DataSet\\VGGDataSet\\SecondEpoche')
    train_losses, test_losses = [], []
    neural_network_model = model_loader.load_classify_model()
    training_service = ts.TrainService(neural_network_model, 0.005, True)
    for epoch in range(epochs):
        train_image_names, test_image_names = utils.split_classify_dataset(utils.get_dataset())
        train_dataset_loader = dsm.DataSetLoader(train_image_names)
        test_dataset_loader = dsm.DataSetLoader(test_image_names)
        train_loss, steps = training_service.train(train_dataset_loader, steps)
        test_loss, accuracy = training_service.eval(test_dataset_loader)
        if steps % print_every == 0:
            message = f"Epoch {epoch + 1}/{epochs} "
            message += f"Train loss: {train_loss/print_every:.3f} "
            message += f"Test loss: {test_loss/len(test_dataset_loader):.3f} "
            message += f"Test accuracy: {accuracy/len(test_dataset_loader):.3f}"
            print(message)
                
        train_losses.append(train_loss/len(test_dataset_loader))
        test_losses.append(test_loss/len(test_dataset_loader))
    
    torch.save(neural_network_model.state_dict(), utils.path_to_output_nn())

Pretrain()
Train()
