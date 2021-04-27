from extract_segmentation import  ExtractSegmentations
import matplotlib.pyplot as plt
import numpy as np

extractor = ExtractSegmentations("../data/val/")
#data = extractor.extractBinaryMasks()
extractor.showBinaryMasks(n=4)




