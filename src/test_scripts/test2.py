import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import os
import PIL
from PIL import ImageOps
import cv2

input_val_dir = "../../data/val/images/"
masks_val_dir = "../../data/val/masks"


input_val_paths = sorted(
    [
        os.path.join(input_val_dir, fname)
        for fname in os.listdir(input_val_dir)
        if fname.endswith(".jpg")
    ]
)

target_val_paths = sorted(
    [
        os.path.join(masks_val_dir, fname)
        for fname in os.listdir(masks_val_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

mask_paths = np.random.choice(target_val_paths, 4)
img_size = (160,160)

for mask in mask_paths:
    print("Using load_img:")
    img = load_img(mask
                   , target_size=img_size
                    , color_mode="grayscale")
    img = img_to_array(img)
    plt.imshow(img)
    plt.show()
    print(np.unique(img))




