import torch
from torch import optim, nn
import Utils

# Service of model training.
class TrainService():
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(model.fc.parameters(), lr = 0.003)
        self.criterion = nn.NLLLoss()
        self.device = Utils.load_device()

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
        loss = 0.0
        self.model.train()
        for trdl_inputs, trdl_labels in dataset_loader.load():
            steps += 1
            inputs, labels = trdl_inputs.to(self.device), trdl_labels.to(self.device)
            self.optimizer.zero_grad()
            logps = self.model.forward(inputs)
            loss = self.criterion(logps, labels)
            loss.backward()
            self.optimizer.step()
            loss += loss.item()
        
        return loss, steps