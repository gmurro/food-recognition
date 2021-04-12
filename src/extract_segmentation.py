from pycocotools.coco import COCO


class ExtractSegmentations:

    def __init__(self, imgDir, annotationPath, masksDir):
        self.imgDir = imgDir
        self.annotationPath = annotationPath
        self.maskDir = masksDir

        self.coco = COCO(annotationPath)


    def getCats(self):
        return self.coco.loadCats(self.coco.getCatIds())


    def mapIdsToIndexes(self):
        ids = self.coco.getCatIds()
        idx = [i for i in range(ids)]
        return idx



