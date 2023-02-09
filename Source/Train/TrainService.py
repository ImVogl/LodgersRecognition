import torch
from torch import optim, nn
import Common.Utils as utilites
import time
from  Common.DiagnisticUtils import Diagnistic
import copy
from torch.optim import lr_scheduler

# Service of model training.
class TrainService():
    # Initialization of train service.
    def __init__(self, model, learning_rate):
        self.imager_per_iteration = 16
        self.diagnostic = Diagnistic()
        self.utils = utilites.Utils()
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr = learning_rate)

        self.criterion = nn.NLLLoss()
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size = 7, gamma = 0.1)
        self.device = self.utils.load_device()

    # Call model forward for test dataset.
    def eval(self, dataset_loader):
        running_loss = 0.0
        running_corrects = 0
        self.model.eval()
        for tdl_inputs, tdl_labels in dataset_loader.load():
            inputs, labels = tdl_inputs.to(self.device), tdl_labels.to(self.device)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    
        epoch_loss = running_loss / len(dataset_loader)
        epoch_acc = running_corrects.double() / len(dataset_loader)

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # deep copy the model       
        best_model_wts = copy.deepcopy(self.model.state_dict())
        torch.save(self.model.state_dict(), './best-model-checkpoint.pt')    
        return self.model, best_model_wts, epoch_acc

    # Call model forward for train dataset.
    def train(self, dataset_loader, steps):
        print("Train process started...")
        start = time.time()
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        for trdl_inputs, trdl_labels in dataset_loader.load():
            steps += 1
            inputs, labels = trdl_inputs.to(self.device), trdl_labels.to(self.device)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            self.scheduler.step()
            print(f"Step: {steps};\tloss: {running_loss:.4f};\telapsed time: {time.time() - start:.2f} seconds.")
        