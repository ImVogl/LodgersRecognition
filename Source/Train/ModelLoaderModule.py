from torch import nn, hub, cuda
from torch import device as torch_device
from torchvision import models

# Loader of model.
class ModelLoader():    
    def __init__(self, model_url):
        state_dict = hub.load_state_dict_from_url(model_url)
        target_device = self.load_device()
        self.neural_network_model = models.resnet50(pretrained = True, progress = False)
        self.neural_network_model.load_state_dict(state_dict)
        for param in self.neural_network_model.parameters():
            param.requires_grad = False

        self.neural_network_model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 10),
                                 nn.LogSoftmax(dim=1))
        self.neural_network_model.to(target_device)

    # Load device type.
    def load_device(self):
        device_name = "cpu"
        if cuda.is_available():
            device_name = "cuda"

        return torch_device(device_name)

    # Load model.
    def load(self):
        return self.neural_network_model
