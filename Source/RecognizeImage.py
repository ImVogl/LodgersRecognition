import Common.Utils as utils
from facenet_pytorch import InceptionResnetV1
from Train.DataSetModule import DataSetLoader
import torch

model = InceptionResnetV1(pretrained='vggface2')
model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 512), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(512, utils.get_last_label()), torch.nn.LogSoftmax(dim = 1))
model.to(utils.load_device())
model.load_state_dict(torch.load(utils.path_to_output_nn()))
model.eval()

def predict_image(image):
    input = image.unsqueeze_(0)
    input.to(utils.load_device())
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

results = []
dataset_loader = DataSetLoader(list(utils.get_all_images('..\\DataSet\\ManualTest', '0')))
for image, label in dataset_loader:
    index = predict_image(image)
    results.append(index)

print(results)