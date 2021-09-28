import os
import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms, dataset

# This function prepares images to train neural network model.
# - It sclales an image to 256x256 points;
# - It cuts an image around a center of the image;
# - It convertes an image to the special torch tensor;
# - It normalizes an image, where 'mean' is average values for each channel, 'std' is standard deviations for each channel.
image_preprocessor = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.255])])

# Loader of images data.
class DataSetLoader(data.Dataset):
    def __init__(self, root, filenames, labels):
        self.root = root
        self.filenames = filenames
        self.labels = labels

    def __getitem__(self, index):
        image_filename = self.filenames[index]
        image_path = Image.open(os.path.join(self.root, image_filename))
        label = self.labels[index]
        
        image = image_preprocessor(image_path)
        label = torch.as_tensor(label, dtype = torch.int64)
        return image, label

    def __len__(self):
        return len(self.filenames)

    # Loading images train data.
    def load(self):
        data_sets = []
        for index in range(self.__len__()):
            data_sets.append(self.__getitem__(index))

        torch_dataset = dataset.TensorDataset(data_sets)
        return data.DataLoader(torch_dataset, batch_size = 64) 
