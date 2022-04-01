import os, shutil
from torch import cuda, nn
from torchsummary import summary
import json
import numpy as np
import Common.Utils as utilites
import matplotlib.pyplot as plt

# Diagnostics class
class Diagnistic():
    def __init__(self):
        utils = utilites.Utils()
        self.path_to_otput = os.path.join(utils.get_working_dir(), '..\\..\\DebugOutput')
        self.model_average_weights_path = os.path.join(self.path_to_otput, 'model_average_weights.json')
        self.model_summary_path = os.path.join(self.path_to_otput, 'model_summary.txt')
    
    # This method showns graphics of average weights for steps of train model
    def show_average_weights(self):
        if not os.path.exists(self.model_average_weights_path):
            return
        
        with open(self.model_average_weights_path, 'r') as model_info_file:
            result = json.load(model_info_file)
        
        plt.title('Average weight on step')
        keys = list(result.keys())
        for index in range(len(result)):
            x = []
            y = []
            for key, value in result[keys[index]].items():
                x.append(int(key))
                y.append(float(value))

            plt.subplot(len(result), 1, index + 1)
            plt.plot(x, y)
            if index + 1 == len(result):
                plt.xlabel('Step number')

            plt.ylabel('Average weight')
            plt.title(keys[index])

        plt.show()

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

    # Save info about nn model weights
    def save_average_weights(self, model, step):
        step = str(step)
        if os.path.exists(self.model_average_weights_path):
            with open(self.model_average_weights_path, 'r') as model_info_file:
                result = json.load(model_info_file)
        else:
            result = {}

        modules = list(model.children())
        for layer in nn.Sequential(*modules):
            if not hasattr(layer, 'weight'):
                continue
            
            layer_name = str(layer).split('(', 1)[0]
            if not (layer_name in result.keys()):
                result[layer_name] = {}
            
            result[layer_name][step] = str(np.average(layer.weight.detach().numpy()))

        layer_name = "Fully connection layer"
        if not (layer_name in result.keys()):
            result[layer_name] = {}

        result[layer_name][step] = str(np.average(model.fc.weight.detach().numpy()))
        with open(self.model_average_weights_path, 'w') as model_info_file:
            json.dump(result, model_info_file)
        
        # Save summary info about model
        def save_summary(self):
            with open(self.model_summary_path, 'w') as model_summary_file:
                model_summary_file.write(summary(model, (3, 256, 256), 16, "cuda" if cuda.is_available() else "cpu"))
            
            