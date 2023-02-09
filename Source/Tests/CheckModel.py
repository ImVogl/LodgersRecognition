import os
import torch
from torchvision import transforms
from PIL import Image

# model=Net() ##Your model here
# pre_trained_path="path.pth"   ##Your saved model path here
# state_dict = torch.load(pre_trained_path)
# model.load_state_dict(state_dict)
# print(f'model {pre_trained_path} loaded')

test_convert_tensor = transforms.Compose([transforms.Resize((64, 128)), transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float)])
target_images_folder = f".\CheckResult"
target_images = {}
for item in os.scandir(target_images_folder):
    target_images[item.path] = int(item.name[0])

device = "cpu"
model = torch.load(f'..\..\PretrainedModels\Result\ResNet18_LodgersRecognition.pt', map_location = torch.device(device))
total = 0
seccesed = 0
with torch.no_grad():
    model.eval()
    for path in target_images.keys():
        target_image = test_convert_tensor(Image.open(path))
        target_image.to(device)
        value, index = torch.max(model(target_image[None, ...]), 1)
        print("Model output: ", int(index) if float(value) > 1 else 3, "; expected output: ", target_images[path])
        total += 1
        if int(index) == target_images[path]:
            seccesed += 1

print(f"Recognized: {seccesed}/{total}; Percent: {(100*seccesed/total):.2f} %")