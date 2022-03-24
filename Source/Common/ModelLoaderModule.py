from torch import nn
from torchvision import models
import Common.Utils as utils
from torchsummary import summary

# Loader of model.
class PretrainedModelLoader():
    def __init__(self):
        utilites = utils.Utils('..\\..\\DataSet\\VGGDataSet\\FirstEpoche')
        target_device = utilites.load_device()
        self.neural_network_model = models.resnet50(pretrained = False, progress = False)
        last_layer = list(self.neural_network_model.children())[-1]
        in_features_prelast_layer = last_layer.in_features
        self.neural_network_model.fc = nn.Linear(in_features_prelast_layer, utilites.get_last_label() + 1)
        for param in self.neural_network_model.parameters():
            param.requires_grad = True

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
        out_features_function = nn.Linear(out_features_prelast_layer, utilites.get_last_label() + 1)
        model.fc = nn.Sequential(in_features_function, nn.ReLU(), nn.Dropout(0.2), out_features_function, nn.LogSoftmax(dim = 1))
            
        return model

    # Load nn info.
    def summary(self):
        print(summary(self.neural_network_model, (3, 256, 256)))
