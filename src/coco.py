from pycocotools.coco import COCO
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("TkAgg")
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
catIds.insert(0,0)
print(len(catIds))


#imgIds = coco.getImgIds(catIds=catIds );

#imgIds = coco.getImgIds(imgIds)

#imgs = coco.loadImgs(imgIds)

#print(imgs)

imgIds = coco.getImgIds()
#print(imgIds)
imgs = coco.loadImgs(imgIds)
print(len(imgs))
'''
for img in imgs:
    anns = coco.getAnnIds(imgIds=img['id'])
    for ann in anns:
        ann = coco.loadAnns(ann)[0]
        print(ann['category_id'])
'''
#print(imgs)

#anns = coco.getAnnIds(imgIds=imgIds)
#print(anns)

img = imgs[9441]
print(img)

original_image = cv2.imread(dataDir+"/"+img['file_name'])
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.show()
original_image = np.array(original_image, dtype=np.uint16)
cv2.imwrite("e.jpg",original_image)

#print(original_image[0,0,:])

'''
anns = coco.getAnnIds(imgIds=img['id'])
print(anns)


mask = np.zeros(original_image.shape[:2])
for ann in anns:
    ann = coco.loadAnns(ann)[0]
    print(ann)
    mask += coco.annToMask(ann)
    #print(np.unique(coco.annToMask(ann)))

    cv2.imwrite(masksDir + "/" + img["file_name"], mask)
    plt.imshow(mask)
    cv2.waitKey(0)
    plt.show()

print(np.unique(mask))
'''
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

'''
def display_sample(display_list):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()
'''