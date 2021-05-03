from pycocotools.coco import COCO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import os

class Segmentation:

    def __init__(self, pathDir, imgDir='images/', annotationFile='annotations.json', masksDir='masks/'):
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

    def getCoco(self):
        return self.coco

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
        """
        Generate a list where each element is an image with multi-channel instance masks.
        Returns:
        masks: A binary array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        imgIds = self.coco.getImgIds()
        imgs = self.coco.loadImgs(imgIds)

        return [self.loadMask(img['id']) for img in imgs]



    def loadMask(self, imgId):
        """
        Generate instance masks for an image.
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
        class_ids = list()

        # Convert annotations to a bitmap mask of shape [height, width, instance_count]
        mask = np.zeros((img['height'], img['width'], instance_count), dtype=np.uint8)

        for i, ann in enumerate(anns):
            ann = self.coco.loadAnns(ann)[0]
            mask[:, :, i] = np.array(self.coco.annToMask(ann), dtype=np.uint8)

            class_ids.append(ann['category_id'])

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        return mask, class_ids



    def extractBinaryMasks(self, saveMasks=True):

        imgIds = self.coco.getImgIds()
        imgs = self.coco.loadImgs(imgIds)

        masks = list()
        print("Masks processing:")
        pbar = tqdm(total=len(imgIds))
        for i, img in enumerate(imgs):
            anns = self.coco.getAnnIds(imgIds=img['id'])

            mask = np.zeros((img['height'], img['width']), dtype=np.uint8)
            for ann in anns:
                ann = self.coco.loadAnns(ann)[0]
                mask = np.bitwise_or(mask, np.array(self.coco.annToMask(ann),  dtype=np.uint8))     # bit or

            if saveMasks:
                file_name = os.path.splitext(img["file_name"])[0] + ".png"
                cv2.imwrite(self.pathDir + self.masksDir + file_name, mask)

            masks.append(mask)

            pbar.update(1)
        return masks



    def showBinaryMasks(self, n=4):
        image_ids = np.random.choice(self.coco.getImgIds(), n)
        imgs_json = self.coco.loadImgs(image_ids)

        for json in imgs_json:
            img = mpimg.imread(self.pathDir+self.imgDir + json['file_name'])
            mask_file_name = os.path.splitext(json['file_name'])[0] + ".png"
            mask = mpimg.imread(self.pathDir+self.masksDir + mask_file_name)

            # plot images
            fig, axes = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)
            ax = axes.flat

            ax[0].imshow(img)
            ax[0].axis('off')

            ax[1].imshow(mask * 255)
            ax[1].axis('off')

            fig.tight_layout()

            plt.show()



    def createSubset(self, filterClasses):

        subset = []

        # Define the classes which you want to see. Others will not be selected.
        subsetImgId = []

        for cl in filterClasses:
            # Fetch class IDs only corresponding to the filterClasses
            catIds = self.coco.getCatIds(catNms=cl)
            # Get all images containing the above Category IDs
            imgIds = self.coco.getImgIds(catIds=catIds)
            print("Number of images containing all the  class {} selected: {}".format(cl, len(imgIds)))
            for id in imgIds:
                subsetImgId.append(id)

        print("Total number of images containing all the classes selected:", len(subsetImgId))
        for i in range(len(subsetImgId)):
            image = self.coco.loadImgs(subsetImgId[i])[0]
            file_name = image['file_name']
            subset.append(file_name)

        return subset



    # Extract mask with a value from 0 to len(labels) for each category in labels
    def extractMasks(self, labels, saveMasks=True):
        imgIds = self.coco.getImgIds()
        imgs = self.coco.loadImgs(imgIds)

        masks = list()

        for img in imgs:
            anns = self.coco.getAnnIds(imgIds=img['id'])

            containsLabel = False

            mask = np.zeros((img['height'], img['width']), dtype=np.uint8)
            for ann in anns:
                ann = self.coco.loadAnns(ann)[0]
                catId = ann['category_id']

                nameCat = self.coco.loadCats(catId)[0]['name']

                if nameCat in labels:
                    containsLabel = True

                    label = labels.index(nameCat)+1
                    mask += (np.array(self.coco.annToMask(ann), dtype=np.uint8) * label)

            if containsLabel:
                masks.append(mask)

                if saveMasks:
                    file_name = os.path.splitext(img["file_name"])[0] + ".png"
                    cv2.imwrite(self.pathDir + self.masksDir + file_name, mask)

        return masks

    """
    # Extract mask with a value from 0 to 273 for each category but it do not work well

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
    """




