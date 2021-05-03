from Segmentation import Segmentation
import shutil
import matplotlib.pyplot as plt
import numpy as np

root = "../data/subset/"
set = "val/"

extractor = Segmentation(root+set)
#data = extractor.extractBinaryMasks()
#extractor.showBinaryMasks(n=4)

sub_root = root+"subset/"
masks = extractor.extractMasks(['banana','broccoli','egg','milk','beer'])

for mask in masks[0:4]:
    plt.imshow(mask)
    plt.show()

"""
subset_img_filename = extractor.createSubset(['banana','broccoli','egg','milk','beer'])
print(subset_img_filename)

sub_root = root+"subset/"
for filename in subset_img_filename:
    shutil.copy(root + set + "images/" + filename, sub_root + set + "images/" + filename)
"""


