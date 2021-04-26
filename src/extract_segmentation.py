from pycocotools.coco import COCO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib

class ExtractSegmentations:

    def __init__(self, pathDir, imgDir='/images/', annotationFile='annotations.json', masksDir='masks/'):
        self.pathDir = pathDir
        self.imgDir = imgDir
        self.annotationFile = annotationFile
        self.masksDir = masksDir
        self.coco = COCO(pathDir+annotationFile)

        # list of category ids adding 0 as first element
        catIds = self.coco.getCatIds()
        catIds.insert(0,0)
        catIds.sort()
        self.catIds = catIds
        self.mapCatIdsLabels = dict(zip(catIds, [i for i in range(len(catIds))]))



    def intToRgb(self, value):
        b = value % 256
        g = ((value - b) / 256) % 256
        r = ((value - b) / 256 ** 2) - g / 256
        return np.array([r,g,b],dtype=np.uint8)

    def rgbToInt(self, pixel):
        r = pixel[0]
        g = pixel[1]
        b = pixel[2]
        return 256**2 * r + 256 * g + b;


    def getMultiChannelMasks(self):
        imgIds = self.coco.getImgIds()
        imgs = self.coco.loadImgs(imgIds)

        masks = list()

        return [self.loadMask(img['id']) for img in imgs]



    def loadMask(self, imgId):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        img = self.coco.loadImgs(imgId)[0]

        # load all annotations for the given image
        anns = self.coco.getAnnIds(imgIds=imgId)
        instance_count = len(anns)


        # array of class IDs of each instance
        catIds = list()

        # Convert annotations to a bitmap mask of shape [height, width, instance_count]
        mask = np.zeros((img['height'], img['width'], instance_count), dtype=np.uint8)

        for i, ann in enumerate(anns):
            ann = self.coco.loadAnns(ann)[0]
            mask[:, :, i] = np.array(self.coco.annToMask(ann), dtype=np.uint8)

            catIds.append(ann['category_id'])

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        return mask, catIds



    def extractMasks(self):
        imgIds = self.coco.getImgIds()
        imgs = self.coco.loadImgs(imgIds)

        masks = list()

        for img in imgs:
            anns = self.coco.getAnnIds(imgIds=img['id'])

            mask = np.zeros((img['height'], img['width']), dtype=np.uint16)
            for ann in anns:
                ann = self.coco.loadAnns(ann)[0]
                catId = ann['category_id']
                mask += (np.array(self.coco.annToMask(ann), dtype=np.uint16)*catId)

            cv2.imwrite(self.pathDir + self.masksDir + img["file_name"], mask)
            masks.append(mask)
        return masks








