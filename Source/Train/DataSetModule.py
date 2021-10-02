import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms
from typing import List
import Model.Image as img_file

# This function prepares images to train neural network model.
# - It sclales an image to 256x256 points;
# - It cuts an image around a center of the image;
# - It convertes an image to the special torch tensor;
# - It normalizes an image, where 'mean' is average values for each channel, 'std' is standard deviations for each channel.
image_preprocessor = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.255])])

# Loader of images data.
class DataSetLoader(data.Dataset):
    def __init__(self, loaded_dataset : List[img_file.TrainImage]):
        self.loaded_dataset = loaded_dataset

    def __getitem__(self, index):
        image = Image.open(self.loaded_dataset[index].image_full_path)
        preprocessed_image = image_preprocessor(image)
        label = torch.as_tensor(self.loaded_dataset[index].label, dtype = torch.int64)
        return preprocessed_image, label

    def __len__(self):
        return len(self.loaded_dataset)

    # Loading images train data.
    def load(self):
        return data.DataLoader(self, batch_size = 64)
