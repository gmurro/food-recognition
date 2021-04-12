from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

dataDir='../data/train/images'
masksDir='../data/train/masks'

coco = COCO("../data/train/annotations.json")


cats = coco.loadCats(coco.getCatIds())
nms=[cat['id'] for cat in cats]
#print(cats)



catIds = coco.getCatIds();

print(len(catIds))


#imgIds = coco.getImgIds(catIds=catIds );

#imgIds = coco.getImgIds(imgIds)

#imgs = coco.loadImgs(imgIds)

#print(imgs)

imgIds = coco.getImgIds()
#print(imgIds)
imgs = coco.loadImgs(imgIds)

#print(imgs)

#anns = coco.getAnnIds(imgIds=imgIds)
#print(anns)

img = imgs[9441]
print(img)

original_image = cv2.imread(dataDir+"/"+img['file_name'])
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.show()

anns = coco.getAnnIds(imgIds=img['id'])
print(anns)

mask = np.zeros(original_image.shape[:2])
for ann in anns:
    ann = coco.loadAnns(ann)[0]
    #print(ann)
    mask += coco.annToMask(ann)
    #print(np.unique(coco.annToMask(ann)))

    cv2.imwrite(masksDir + "/" + img["file_name"], mask)
    plt.imshow(mask)
    cv2.waitKey(0)
    plt.show()

print(np.unique(mask))
"""
for img in imgs:
    ann = coco.getAnnIds(imgIds=img['id'])

    ann = coco.loadAnns(ann)[0]
    # print(ann)
    m = coco.annToMask(ann) * 255
    # print(m)
    cv2.imwrite(masksDir + "/" + img["file_name"], m)
    #cv2.imshow("", m)
    #cv2.waitKey(0)
    
"""
