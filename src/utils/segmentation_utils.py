from pycocotools.coco import COCO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import os


def load_coco_dataset(pathDir, annotationFile='annotations.json'):
    return COCO(pathDir + annotationFile)


def augmentation(images, masks):
    return None
    # return augmented (images, masks)
