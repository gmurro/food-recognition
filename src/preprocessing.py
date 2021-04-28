from Segmentation import Segmentation

extractor = Segmentation("../data/train/")
data = extractor.extractBinaryMasks()
extractor.showBinaryMasks(n=4)




