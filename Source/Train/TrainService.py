import torch
from torch import optim, nn
import Common.Utils as utilites
import time
from  Common.DiagnisticUtils import Diagnistic

# Service of model training.
class TrainService():
    # Initialization of train service.
    def __init__(self, model, learning_rate, only_last_layer: bool = False):
        self.imager_per_iteration = 16
        self.diagnostic = Diagnistic()
        self.utils = utilites.Utils()
        self.model = model
        self.only_last_layer = only_last_layer
        if only_last_layer:
            self.optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
            self.freeze_layers()
        else:
            self.optimizer = optim.Adam(model.parameters(), lr = learning_rate)
            for param in self.model.parameters():
                param.requires_grad = True

        self.criterion = nn.NLLLoss()
        self.device = self.utils.load_device()

    # Call model forward for test dataset.
    def eval(self, dataset_loader):
        loss = 0.0
        accuracy = 0.0
        self.model.eval()
        with torch.no_grad():
            for tdl_inputs, tdl_labels in dataset_loader.load():
                inputs, labels = tdl_inputs.to(self.device), tdl_labels.to(self.device)
                logps = self.model.forward(inputs)
                batch_loss = self.criterion(logps, labels)
                loss += batch_loss.item()
                        
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim = 1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        return loss, accuracy

    # Call model forward for train dataset.
    def train(self, dataset_loader, steps):
        print("Train process started...")
        start = time.time()
        loss = 0.0
        self.model.train()
        for trdl_inputs, trdl_labels in dataset_loader.load():
            steps += 1
            inputs, labels = trdl_inputs.to(self.device), trdl_labels.to(self.device)
            self.optimizer.zero_grad()
            logps = self.convert_logs(self.model.forward(inputs))
            loss = self.criterion(logps, labels)
            if not self.only_last_layer:
                loss.backward()

            self.optimizer.step()
            loss += loss.item()
            print(f"Step: {steps};\tloss: {loss:.4f};\telapsed time: {time.time() - start:.2f} seconds.")
            if (steps - 1) % 20 == 0:
                self.diagnostic.save_average_weights(self.model, steps - 1)
        
        return loss, steps
    
    # Преобразование тензора с убывающей ошибкой.
    def convert_logs(self, logps):
        result_tensor = torch.zeros(len(logps), len(logps[0]), dtype = logps.dtype)
        for i in range(len(logps)):
            for j in range(len(logps[0])):
                result_tensor[i][j] = self.extract_data(logps[i][j], logps.dtype)
        
        return result_tensor

    # Извлечение значения.
    def extract_data(self, item, type):
        if item.ndim > 1:
            return self.extract_data(item[0], type)
        return item

    # Freeze layers till full connected layer.
    def freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.fc.parameters():
            param.requires_grad = True
        
        self.model
