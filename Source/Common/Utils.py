import os, shutil
import Train.Model.Image as img_file
import imghdr
from typing import List
import numpy as np
from torch import device as torch_device, cuda, layer_norm, nn
from torchsummary import summary
import json

# Support utils for recognition and training
class Utils():
    def __init__(self, dataset : str = None):
        # Path to folder with data set
        if dataset == None:
            self.data_set_base_path = '..\\..\\DataSet\\Current'
        else:
            self.data_set_base_path = dataset

        self.path_to_otput = os.path.join(self.get_working_dir(), '..\\..\\DebugOutput')

    # Get path to script working folder.
    def get_working_dir(self):
        return os.path.dirname(os.path.realpath(__file__))

    # Getting full path to pretrained neural network file.
    def get_resnet50_full_path(self):
        resnet50_network_file_name = '20180408-102900-casia-webface.pt'
        return os.path.join(self.get_working_dir(), '..\\..\\PretrainedModels', resnet50_network_file_name)

    # Getting preparated dataset.
    def get_dataset(self):
        result = []
        root_folder = os.path.join(self.get_working_dir(), self.data_set_base_path)
        all_dir_items = os.listdir(root_folder)
        for item in all_dir_items:
            full_path = os.path.join(root_folder, item)
            if not os.path.isdir(full_path):
                continue

            result += list(self.get_all_images(root_folder, item))
        return result

    # Getting all images for target label.
    def get_all_images(self, root: str, subfolder: str):
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
    def split_dataset(self, data_set: List[img_file.TrainImage], test_proportion: float = 0.2):
        np.random.shuffle(data_set)
        num_train = len(data_set)
        split_position = int(np.floor(test_proportion * num_train))
        return data_set[split_position:], data_set[:split_position]

    # Splitting of dataset to train and test collection for classification.
    def split_classify_dataset(self, data_set: List[img_file.TrainImage]):
        test_proportion = 1/self.get_last_label()
        data_set_by_labels = {}
        for item in data_set:
            if item.label in data_set_by_labels.keys:
                data_set_by_labels[item.label].append(item)
            else:
                data_set_by_labels[item.label] = [item]

        minimum_images_count = len(data_set_by_labels.values[0])
        for key in data_set_by_labels.keys:
            if minimum_images_count < len(data_set_by_labels[key]):
                minimum_images_count = len(data_set_by_labels[key])

        for key in data_set_by_labels.keys:
            np.random.shuffle(data_set_by_labels[key])

        images_per_label_count = int(np.floor(test_proportion * minimum_images_count))
        result_test_dataset = []
        result_train_data_set = []
        for value in data_set_by_labels.values:
            result_test_dataset.append(value[:images_per_label_count])
            result_train_data_set.append(value[images_per_label_count:])

        return result_train_data_set, result_test_dataset

    # Getting file with trained neural network. 
    def path_to_output_nn(self):
        model_file_name = 'ResNet50_LodgersRecognition.pt'
        result_dir = os.path.join(self.get_working_dir(), '..\\..\\PretrainedModels\\Result')
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)

        return os.path.join(result_dir, model_file_name)

    # Getting maximal label value.
    def get_last_label(self):
        max_label = -1
        root_folder = os.path.join(self.get_working_dir(), self.data_set_base_path)
        all_dir_items = os.listdir(root_folder)
        for item in all_dir_items:
            full_path = os.path.join(root_folder, item)
            if not (os.path.isdir(full_path) and item.isdigit()):
                continue
            
            int_item = int(item)
            if max_label < int_item:
                max_label = int_item
            
        return max_label

    # Load device type.
    def load_device(self):
        device_name = "cpu"
        if cuda.is_available():
            device_name = "cuda"

        return torch_device(device_name)
    
    # Cleaning folder with debug info.
    def clean_debug_folder(self):
        for filename in os.listdir(self.path_to_otput):
            file_path = os.path.join(self.path_to_otput, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
            

    # Save info about nn model
    def summary(self, model):
        model_info_path = os.path.join(self.path_to_otput, 'model.json')
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as model_info_file:
                result = json.load(model_info_file)
        else:
            result = {}

        modules = list(model.children())
        for layer in nn.Sequential(*modules):
            if not hasattr(layer, 'weight'):
                continue
            
            layer_name = str(layer).split('(', 1)[0]
            if not (layer_name in result.keys()):
                result[layer_name] = []
            
            result[layer_name].append(str(np.average(layer.weight.detach().numpy())))

        layer_name = "Fully connection layer"
        if not (layer_name in result.keys()):
            result[layer_name] = []

        result[layer_name].append(str(np.average(model.fc.weight.detach().numpy())))
        with open(model_info_path, 'w') as model_info_file:
            json.dump(result, model_info_file)
        
        # info_writer.writerow(["Summary info"])
        # info_writer.writerows(summary(model, (3, 256, 256), 16, "cuda" if cuda.is_available() else "cpu"))
