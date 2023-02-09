from torch import nn
from torchvision import models
import Common.Utils as utils

# Loader of model.
class PretrainedModelLoader():
    def __init__(self):
        utilites = utils.Utils('..\\..\\DataSet\\VGGDataSet\\FirstEpoche')
        target_device = utilites.load_device()
        self.neural_network_model = models.resnet18(pretrained = True, progress = False)
        in_features_prelast_layer = self.neural_network_model.fc.in_features
        self.neural_network_model.fc = nn.Linear(in_features_prelast_layer, utilites.get_last_label())
        self.neural_network_model.to(target_device)

    # Save model
    def save(self, model):
        self.neural_network_model = model

    # Load model.
    def load(self):
        return self.neural_network_model
