import Common.Utils as utils
from facenet_pytorch import InceptionResnetV1
from Train.DataSetModule import DataSetLoader
import torch

from torchsummary import summary

model = torch.nn.Sequential(InceptionResnetV1(pretrained='vggface2'), torch.nn.Linear(512, utils.get_last_label() + 1))
model.to(utils.load_device())
print(summary(model, (3, 224, 224), batch_size = 1, device = "cpu"))
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