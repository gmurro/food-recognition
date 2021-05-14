from Segmentation import Segmentation
import shutil
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
root = "../data/"
set = "train/"

extractor = Segmentation(root+set)
distrCats = extractor.distributionClasses(18)
total = 0
for i,cat in enumerate(distrCats):
    print("{}. {} : {}".format(i+1,cat['name'],cat['count']))
    total += cat['count']
print("Tot:", total)

subset_img_filename = extractor.createSubset(['breadwhite'])
for filename in subset_img_filename[20:30]:
    path = root + set + "images/" + filename
    img = img_to_array(load_img(path))/255
    plt.imshow(img)
    plt.title("bread-white")
    plt.show()


subset_img_filename = extractor.createSubset(['breadwholemeal'])
for filename in subset_img_filename[20:23]:
    path = root + set + "images/" + filename
    img = img_to_array(load_img(path))/255
    plt.imshow(img)
    plt.title("bread-wholemeal")
    plt.show()



#data = extractor.extractBinaryMasks()
#extractor.showBinaryMasks(n=4)
"""
sub_root = root+"subset/"
masks = extractor.extractMasks(['banana','broccoli','egg','milk','beer'])

for mask in masks[0:4]:
    print(np.unique(mask))
    plt.imshow(mask)
    plt.show()


subset_img_filename = extractor.createSubset(['banana','broccoli','egg','milk','beer'])
print(subset_img_filename)

sub_root = root+"subset/"
for filename in subset_img_filename:
    shutil.copy(root + set + "images/" + filename, sub_root + set + "images/" + filename)
"""


