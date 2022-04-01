# This script file shows image pairs before and after normalization
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Common import Utils as utilites
from Train import DataSetModule as dsm
import tkinter as tk
from PIL import ImageTk, Image
import torchvision.transforms as tv

transform = tv.ToPILImage()
utils = utilites.Utils('..\\..\\DataSet\\VGGDataSet\\FirstEpoche')
dataset = utils.get_dataset()
dataset_loader = dsm.DataSetLoader(dataset)
length = len(dataset)
current_index = 0

window = tk.Tk()
window.title("Images comparison")
window.geometry("500x250")
window.configure(background='grey')

panel_with_source = tk.Label(window, text = 'Before preprocessing')
panel_with_processed = tk.Label(window, text = 'After preprocessing')

# Show images in window
def show_images():
    source_image = ImageTk.PhotoImage(Image.open(dataset[current_index].image_full_path))
    processed_image = ImageTk.PhotoImage(transform(dataset_loader[current_index][0]))
    panel_with_source.configure(image = source_image)
    panel_with_source.image = source_image
    panel_with_processed.configure(image = processed_image)
    panel_with_processed.image = processed_image

show_images()
panel_with_source.pack(side = "left", fill = "both", expand = "yes")
panel_with_processed.pack(side = "right", fill = "both", expand = "yes")

# Left key processing.
def key_left(event):
    global current_index
    if current_index > 1:
        current_index -= 1
    else:
        current_index = length - 1
    
    show_images()

# Right key processing.
def key_right(event):
    global current_index
    if current_index < length - 1:
        current_index += 1
    else:
        current_index = 0
    
    show_images()

window.bind('<Left>',key_left)
window.bind('<Right>',key_right)
window.mainloop()
