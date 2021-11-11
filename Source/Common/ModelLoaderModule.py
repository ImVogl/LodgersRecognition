from torch import nn, optim
from torchvision import models
import Common.Utils as utils

# Loader of model.
class PretrainedModelLoader():
    def __init__(self):
        utilites = utils.Utils()
        target_device = utilites.load_device()
        model = models.resnet50(pretrained = True, progress = False)
        self.neural_network_model = nn.Sequential(*(list(model.children())[:-1]))
        for param in self.neural_network_model.parameters():
            param.requires_grad = False

        for group in optim.param_groups:
            group['lr'] = 0.0005

        self.neural_network_model.to(target_device)

    # Save model
    def save(self, model):
        self.neural_network_model = model

    # Load model.
    def load(self):
        return self.neural_network_model

    # Load model for classification.
    def load_classify_model(self):
        utilites = utils.Utils('..\\..\\DataSet\\VGGDataSet\\FirstEpoche')
        model = self.neural_network_model
        last_layer = list(model.children())[-1]
        in_features_prelast_layer = last_layer.in_features
        out_features_prelast_layer = utilites.get_last_label()
        in_features_function = nn.Linear(in_features_prelast_layer, out_features_prelast_layer)
        utilites = utils.Utils('..\\..\\DataSet\\VGGDataSet\\SecondEpoche')
        out_features_function = nn.Linear(out_features_prelast_layer, utilites.get_last_label())
        model.fc = nn.Sequential(in_features_function, nn.ReLU(), nn.Dropout(0.2), out_features_function, nn.LogSoftmax(dim = 1))
        optim.param_groups[-1]['lr'] = 0.005
            
        return model