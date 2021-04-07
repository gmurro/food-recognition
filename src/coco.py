from pycocotools.coco import COCO
import cv2

dataDir='..\\data\\train\\images'
masksDir='..\\data\\train\\masks'

coco = COCO("..\\data\\train\\annotations.json")

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]

print(nms)
catIds = coco.getCatIds(catNms=nms);

#print(catIds)

imgIds = coco.getImgIds(catIds=catIds );

imgIds = coco.getImgIds(imgIds)

imgs = coco.loadImgs(imgIds)

#print(imgs)

imgs = coco.loadImgs(imgIds)

#anns = coco.getAnnIds(imgIds=imgIds)
#print(anns)

for img in imgs:
    ann = coco.getAnnIds(imgIds=img['id'])

    ann = coco.loadAnns(ann)[0]
    # print(ann)
    m = coco.annToMask(ann) * 255
    # print(m)
    cv2.imwrite(masksDir + "/" + img["file_name"], m)
    #cv2.imshow("", m)
    #cv2.waitKey(0)
    
    
