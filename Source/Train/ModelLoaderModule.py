from torch import nn, hub
from torch import load as torch_load
from torchvision import models
import Utils
from facenet_pytorch import InceptionResnetV1

# Loader of model.
class ModelLoader():    
    def __init__(self, model_url):
        state_dict = hub.load_state_dict_from_url(model_url)
        self.internal_init(state_dict)

    def __init__(self, from_file: bool = False):
        if from_file:
            self.internal_init(Utils.get_resnet50_full_path())
        else:
            self.init_with_module()

    # Internal initialization of model loader.
    def internal_init(self, path):
        target_device = Utils.load_device()
        self.neural_network_model = models.resnet50(pretrained = True, progress = False)
        self.neural_network_model.load_state_dict(torch_load(path, map_location = target_device))
        for param in self.neural_network_model.parameters():
            param.requires_grad = False

        self.neural_network_model.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, Utils.get_last_label()), nn.LogSoftmax(dim = 1))
        self.neural_network_model.to(target_device)

    # Initializing neural network with module from https://modelzoo.co/model/facenet-pytorch
    def init_with_module(self):
        self.neural_network_model = InceptionResnetV1(pretrained='vggface2')
        for param in self.neural_network_model.parameters():
            param.requires_grad = True

        self.neural_network_model.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, Utils.get_last_label()), nn.LogSoftmax(dim = 1))
        self.neural_network_model.to(Utils.load_device())

    # Load model.
    def load(self):
        return self.neural_network_model
