from pycocotools.coco import COCO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import os
from skimage.transform import resize

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import Progbar


def load_coco_dataset(pathDir, annotationFile='annotations.json'):
    return COCO(os.path.join(pathDir, annotationFile))


def augmentation(images, masks):
    return None
    # return augmented (images, masks)


def load_data(coco, imgDir, img_size, cat_names=[],  show_progress=True):

    # load images info from annotations
    img_ids, img_names, cat_names, cat_map = load_imgs(coco, cat_names)

    # prepare structure of input images
    x = np.zeros((len(img_names),) + img_size + (3,), dtype="float32")

    if show_progress:
        print("Loading images:")
        pbar = Progbar(target=len(img_names))

    for i, filename in enumerate(img_names):
        path = os.path.join(imgDir, filename)
        img = img_to_array(load_img(path, target_size=img_size))
        x[i] = img

        if show_progress:
            pbar.update(i)

    # prepare structure of target masks
    y = np.zeros((len(img_names),) + img_size + (len(cat_names),), dtype="float32")

    if show_progress:
        print("Loading masks:")
        pbar = Progbar(target=len(img_names))

    for i, img_id in enumerate(img_ids):

        mask = load_mask(coco, img_id, cat_map)
        # TODO resize mask
        y[i] = mask
        if show_progress:
            pbar.update(i)
    return x


def load_imgs(coco, cat_names=[]):
    """
    Load image names and filter the dataset according to the list of categories specified.

    Parameters
    ----------
    coco: object
        COCO object containing info about the COCO annotations
    cat_names: list, optional
        List of categories according to filter the dataset. If empty, load all dataset

    Returns
    -------
    tuple
        Tuple composed by:
         - a list of image ids
         - a list of  image names
         - the new list of categories ordered and with the special category 'background' added
         - a dict that maps category ids to the index of the previous array
    """

    # get cat ids
    if cat_names is None:
        cat_names = []
    cat_ids = coco.getCatIds(catNms=cat_names)

    # get cat_names ordered and with the special category 'background'
    cats = coco.loadCats(cat_ids)
    cat_names = ['background']+[cat['name'] for cat in cats]

    # build a map from cat_ids to to the corresponded index of cat_names
    cat_map = dict(zip([0] + cat_ids, range(len(cat_names))))

    # get images annotated with the filtered categories
    img_ids = [coco.getImgIds(catIds=[id]) for id in cat_ids]
    img_ids = [item for sublist in img_ids for item in sublist]  # List of lists flattening
    img_names = [coco.loadImgs(img_id)[0]['file_name'] for img_id in img_ids]

    return img_ids, img_names, cat_names, cat_map


def load_mask(coco, img_id, cat_map):
    """
    Generate a multichannel mask for an image.

    Parameters
    ----------
    coco: object
        coco object containing info about the COCO annotations
    img_id: int
        id of the image for which you want get the mask
    cat_map: dict
        Map from cat_ids to to the corresponded index of cat_names

    Returns
    -------
    A np.array of shape [height, width, n_categories] where each channel is a binary mask correspondent to each category.
    The first channel is dedicated to the background
    """

    img = coco.loadImgs(img_id)[0]

    # load all annotations for the given image
    anns = coco.getAnnIds(imgIds=img_id)

    # number of categories according to generate the mask. The background must be included
    n_categories = len(cat_map)

    # Convert annotations to a binary mask of shape [height, width, n_categories]
    mask = np.zeros((img['height'], img['width'], n_categories), dtype=np.float32)

    # channel dedicated to the background
    background = np.zeros((img['height'], img['width']), dtype=np.uint8)

    for ann in anns:
        ann = coco.loadAnns(ann)[0]

        # check if it is a considered category
        if ann['category_id'] in list(cat_map.keys()):

            # get channel to fill
            i = cat_map[ann['category_id']]
            mask[:, :, i] = np.array(coco.annToMask(ann), dtype=np.float32)

            # compose incrementally the background channel
            background = np.bitwise_or(background, coco.annToMask(ann))

    # invert background to set the background as 1 wherever all other mask are 0
    background = np.logical_not(background).astype(int)
    mask[:, :, 0] = np.array(background, dtype=np.float32)

    return mask




