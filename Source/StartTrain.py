import Train.DataSetModule as dsm
import Common.ModelLoaderModule as mlm
import torch
import Common.Utils as utilites
import Train.TrainService as ts
import time
import copy

# The pretrained network was got from
# https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/torchvision_models.py
# https://github.com/pytorch/vision/tree/master/torchvision/models
# https://download.pytorch.org/models/resnet50-19c8e357.pth
# https://github.com/akhiilkasare/MRI-Brain-Tumor-Classification-using-Pytorch/blob/e0a6adbe281ff4f31172cc8c90352f7332680bc6/main.py#L40

# Setting up
model_loader = mlm.PretrainedModelLoader()

# Train model.
def Train():
    since = time.time()
    epochs = 6
    steps = 0
    utils = utilites.Utils('..\\..\\DataSet\Current\Cut')
    neural_network_model = model_loader.load()
    best_model_wts = copy.deepcopy(neural_network_model.state_dict())
    best_acc = 0.0
    training_service = ts.TrainService(neural_network_model, 0.0002)
    for epoch in range(epochs):
        train_image_names, test_image_names = utils.split_dataset(utils.get_dataset())
        train_dataset_loader = dsm.DataSetLoader(train_image_names)
        test_dataset_loader = dsm.DataSetLoader(test_image_names)
        training_service.train(train_dataset_loader, steps)
        neural_network_model, best_model_wts, best_acc = training_service.eval(test_dataset_loader)
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    neural_network_model.load_state_dict(best_model_wts)
    torch.save(neural_network_model, utils.path_to_output_nn())

Train()
