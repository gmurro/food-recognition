from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

dataDir='..\\data\\train\\images'
masksDir='..\\data\\train\\masks'

coco = COCO("..\\data\\train\\annotations.json")
img = Image.open("C:/Users/peppe/UNIBO/Deep Learning/lab/FoodRecognition/data/train/images/006316.jpg")
plt.imshow(img)


fig, ax = plt.subplots()
ax.imshow(img)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
plt.show()

cats = coco.loadCats(coco.getCatIds())
nms=[cat['id'] for cat in cats]
#print(cats)


catIds = coco.getCatIds();

print(catIds)


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

img = imgs[0]
print(img)
ann = coco.getAnnIds(imgIds=img['id'])
ann = coco.loadAnns(ann)[0]
# print(ann)
m = coco.annToMask(ann) #* 255
print(np.unique(coco.annToMask(ann)))
cv2.imwrite(masksDir + "/" + img["file_name"], m)
plt.imshow(m)
cv2.waitKey(0)

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
