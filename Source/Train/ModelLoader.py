from torch import nn, hub, cuda, optim
from torch import device as torch_device

# Loader of model.
class ModelLoader():
    neural_network_model = None
    def __init__(self, model_url, model_folder):
        state_dict = hub.load_state_dict_from_url(model_url)
        device_name = "cpu"
        if cuda.is_available():
            device_name = "cuda"

        target_device = torch_device(device_name)
        neural_network_model = models.resnet50(pretrained = True, model_dir = model_folder, file_name = 'resnet50.pth', progress = False)
        neural_network_model.load_state_dict()
        for param in neural_network_model.parameters():
            param.requires_grad = False

        neural_network_model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 10),
                                 nn.LogSoftmax(dim=1))
        neural_network_model.to(torch_device)

    # Load model    
    def load(self):
        return neural_network_model
