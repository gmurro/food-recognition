
from pycocotools.coco import COCO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
from random import random
import os
from skimage.transform import resize

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import Progbar
from keras.preprocessing.image import ImageDataGenerator


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
    cat_names = ['background' ] +[cat['name'] for cat in cats]

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


def get_augmented(X, y, image_size, num_classes, batch_size=16, index=-1):
    """
  Generative function that create a batch of augumented images and masks
  for the passed set of images and their correspondent masks.

  Parameters
  ----------
  X :nparray
      set of images
  y :nparray
        set of masks with index equal to the one of the corresponding image
  image_size : tuple of int size 2
         size of the image and mask, first parameter is the weight, the second is the height
  num_classes: int
         number of categories for the segmentation
  batch_size: int
         size of the batch created for each image
  index: int
         index of the image to generate the batch, if not present it will choice randomly
  Returns
  -------
    Two np.array, one that represent the batch of modified images with shape [batch,sizeheight, width, n_categories]
    and the other one that represent the batch modified masks with shape [batch,sizeheight, width, n_categories]
    where last channel is the binary mask correspondent to each category.
  """
    # the arguments used by keras ImageGenerator for images
    aug_generator_args = dict(featurewise_center=False,
                              samplewise_center=False,
                              rotation_range=5,
                              width_shift_range=0.01,
                              height_shift_range=0.01,
                              brightness_range=(0.8, 1.2),
                              shear_range=0.01,
                              zoom_range=[1, 1.25],
                              horizontal_flip=True,
                              vertical_flip=False,
                              fill_mode='reflect',
                              data_format='channels_last')

    aug_generator_args_mask = aug_generator_args.copy()
    # the arguments used by keras ImageGenerator for mask (same but without brightness)
    aug_generator_args_mask.pop('brightness_range' ,None)

    image_gen = ImageDataGenerator(**aug_generator_args)
    mask_gen = ImageDataGenerator(**aug_generator_args_mask)

    while True:

        if index == -1:
            idx = np.random.choice(len(X - 1))
        else:
            idx = index

        image = X[idx]
        mask = y[idx]
        print(mask.shape)
        valid_mask = []
        for i in range(num_classes):
            channel = mask[:, :, i]
            if channel.any():
                valid_mask.append(channel)

        X_aug = np.zeros((batch_size, image_size[0], image_size[1], 3)).astype('float')
        y_aug = np.zeros((batch_size, image_size[0], image_size[1], num_classes)).astype('float')

        seed = np.random.choice(range(9999))

        g_x = image_gen.flow(image.reshape((1,) + image.shape),
                             batch_size=batch_size,
                             seed=seed)

        g_ys = []

        for i in range(len(valid_mask)):
            g_ys.append(mask_gen.flow(valid_mask[i].reshape((1,) + valid_mask[i].shape + (1,)),
                                      batch_size=batch_size,
                                      seed=seed))

        for batch_num in range(batch_size):
            X_aug[batch_num] = g_x.next() / 255.0
            for i in range(len(valid_mask)):
                cat_channel = g_ys[i].next().reshape([160, 160])
                y_aug[batch_num, :, :, i] = cat_channel

        yield X_aug, y_aug
