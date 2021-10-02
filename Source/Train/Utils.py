import os
import Model.Image as img_file
import imghdr
from typing import List
import numpy as np
from torch import device as torch_device, cuda

# Get path to script working folder.
def get_working_dir():
    return os.path.dirname(os.path.realpath(__file__))

# Getting full path to pretrained neural network file.
def get_resnet50_full_path():
    resnet50_network_file_name = '20180408-102900-casia-webface.pt'
    return os.path.join(get_working_dir(), '..\\..\\PretrainedModels', resnet50_network_file_name)

# Getting preparated dataset.
def get_dataset():
    result = []
    root_folder = os.path.join(get_working_dir(), '..\\..\\DataSet\\Current')
    all_dir_items = os.listdir(root_folder)
    for item in all_dir_items:
        full_path = os.path.join(root_folder, item)
        if not os.path.isdir(full_path):
            continue

        result += list(get_all_images(root_folder, item))
    return result

# Getting all images for target label.
def get_all_images(root: str, subfolder: str):
    if not subfolder.isdigit():
        return

    for name in os.listdir(os.path.join(root, subfolder)):
        full_path = os.path.join(root, subfolder, name)
        if imghdr.what(full_path) != None:
            image = img_file.TrainImage()
            image.image_full_path = full_path
            image.image_name = name
            image.label = int(subfolder)
            yield image

# Splitting of dataset to train and test collection.
def split_dataset(data_set: List[img_file.TrainImage], test_proportion: float = 0.2):
    np.random.shuffle(data_set)
    num_train = len(data_set)
    split_position = int(np.floor(test_proportion * num_train))
    return data_set[split_position:], data_set[:split_position]

# Getting file with trained neural network. 
def path_to_output_nn():
    model_file_name = 'ResNet50_LodgersRecognition.pth'
    result_dir = os.path.join(get_working_dir(), '..\\..\\PretrainedModels\\Result')
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    return os.path.join(result_dir, model_file_name)

# Getting maximal label value.
def get_last_label():
    max_label = -1
    root_folder = os.path.join(get_working_dir(), '..\\..\\DataSet\\Current')
    all_dir_items = os.listdir(root_folder)
    for item in all_dir_items:
        full_path = os.path.join(root_folder, item)
        int_item = int(item)
        if os.path.isdir(full_path) and item.isdigit() and max_label < int_item:
            max_label = int_item
        
    return max_label

# Load device type.
def load_device():
    device_name = "cpu"
    if cuda.is_available():
        device_name = "cuda"

    return torch_device(device_name)
