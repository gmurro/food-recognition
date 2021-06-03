from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import os
from PIL import Image

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import Progbar
from keras.preprocessing.image import ImageDataGenerator


def load_coco_dataset(path_dir, annotation_file='annotations.json'):
    """
    Load COCO object reading info about the COCO annotations

    Parameters
    ----------
    path_dir : str
        The directory containing the annotations file
    annotation_file : str
        Filename of the annotations file

    Returns
    -------
    object
        COCO object containing info about the COCO annotations
    """
    return COCO(os.path.join(path_dir, annotation_file))


def load_data(coco, img_dir, img_size, cat_names=[],  show_progress=True):
    """
    Load the whole dataset in one shot, filtering it for some categories

    Parameters
    ----------
    coco: object
        COCO object containing info about the COCO annotations
    img_dir: str
        The path to the directory containing images to read
    img_size: tuple
        Height and width according to resize the images
    cat_names: list, optional
        List of categories according to filter the dataset. If empty, load all dataset
    show_progress: bool, optional
        If True, it allows to show the progress of the loading

    Returns
    ----------
    tuple
        (x, y) where the first element is a np.array of input image
        and the second element is a np.array of binary masks
    """

    # load images info from annotations
    img_ids, img_names, cat_names, cat_to_label = load_imgs(coco, cat_names)

    # prepare structure of input images
    x = np.zeros((len(img_names),) + img_size + (3,), dtype=np.float32)

    if show_progress:
        print("Loading images:")
        bar_img = Progbar(target=len(img_names))

    for i, filename in enumerate(img_names):
        path = os.path.join(img_dir, filename)
        img = img_to_array(load_img(path, target_size=img_size))
        x[i] = img

        if show_progress:
            bar_img.update(i)

    # prepare structure of target masks
    y = np.zeros((len(img_names),) + img_size + (len(cat_names),), dtype=np.float32)

    if show_progress:
        print("\nLoading masks:")
        bar_mask = Progbar(target=len(img_ids))

    for i, img_id in enumerate(img_ids):
        mask = load_mask(coco, img_id, cat_to_label)

        # scale down mask
        mask_scaled = np.zeros(img_size + (len(cat_names),), dtype=np.uint8)
        for j in range(len(cat_names)):
            mask_scaled[:, :, j] = Image.fromarray(mask[:, :, j]).resize(img_size, Image.NEAREST)

        y[i] = np.array(mask_scaled, dtype=np.float32)

        if show_progress:
            bar_mask.update(i)
    return x, y


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
    cat_names = ['background'] +[cat['name'] for cat in cats]

    # build a map from cat_ids to to the corresponded index of cat_names
    cat_to_label = dict(zip([0] + cat_ids, range(len(cat_names))))

    # get images annotated with the filtered categories
    img_ids = [coco.getImgIds(catIds=[id]) for id in cat_ids]
    img_ids = [item for sublist in img_ids for item in sublist]  # List of lists flattening
    img_names = [coco.loadImgs(img_id)[0]['file_name'] for img_id in img_ids]

    return img_ids, img_names, cat_names, cat_to_label


def load_mask(coco, img_id, cat_to_label):
    """
    Generate a multichannel mask for an image.

    Parameters
    ----------
    coco: object
        coco object containing info about the COCO annotations
    img_id: int
        id of the image for which you want get the mask
    cat_to_label: dict
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
    n_categories = len(cat_to_label)

    # Convert annotations to a binary mask of shape [height, width, n_categories]
    mask = np.zeros((img['height'], img['width'], n_categories), dtype=np.float32)

    # channel dedicated to the background
    background = np.zeros((img['height'], img['width']), dtype=np.uint8)

    for ann in anns:
        ann = coco.loadAnns(ann)[0]

        # check if it is a considered category
        if ann['category_id'] in list(cat_to_label.keys()):

            # get channel to fill
            i = cat_to_label[ann['category_id']]
            mask[:, :, i] = np.array(coco.annToMask(ann), dtype=np.float32)

            # compose incrementally the background channel
            background = np.bitwise_or(background, coco.annToMask(ann))

    # invert background to set the background as 1 wherever all other mask are 0
    background = np.logical_not(background).astype(int)
    mask[:, :, 0] = np.array(background, dtype=np.float32)

    return mask


def augment(x, y, img_aug_args):
    """
    Generative function that create a batch of augmented images and masks
    for the passed set of images and their correspondent masks.

    Parameters
    ----------
    x : np.array
        Batch of input images
    y : np.array
        Batch of masks
    img_aug_args: dict
        Parameters of the Keras ImageDataGenerator class
    Returns
    ----------
    tuple
        (x, y) of the same shape of the input ones, but transformed
    """

    # the brightness parameter is unuseful for the mask, remove it
    if img_aug_args.keys().__contains__('brightness_range'):
        img_aug_args.pop('brightness_range', None)

    # we apply data augmentation to our datasets using Keras ImageDataGenerator class
    image_aug = ImageDataGenerator(**img_aug_args)

    # transpose the batch of the masks to iter over the channels
    y_t = y.transpose((3, 0, 1, 2))

    # prepare data structure for the augmentation result
    x_aug = np.zeros(x.shape, dtype=np.float32)
    y_aug = np.zeros(y_t.shape, dtype=np.float32)

    # seed useful to sync the augmentation fof x and y applying the same transformation
    seed = np.random.choice(range(9999))

    batch_size = x.shape[0]
    n_channel = y.shape[3]

    # create a generator that can return an augmented batch of images
    image_gen = image_aug.flow(x, batch_size=batch_size, shuffle=False, seed=seed)

    mask_gen = []
    for i in range(n_channel):
        # create a generator that can return an augmented batch of masks
        mask_gen.append(image_aug.flow(y_t[i].reshape(y_t[i].shape + (1,)), shuffle=False, batch_size=batch_size, seed=seed))

    # get the augmented batch
    x_aug = image_gen.next()
    for i in range(n_channel):
        y_aug[i] = mask_gen[i].next()[:,:,:,0]

    return x_aug, np.round(y_aug.transpose((1, 2, 3, 0)))


# TODO Modify load_imagse since it retuns duplicates
def get_class_weight(coco, cat_names=[]):
    """
    Compute weights for the given category on the training set

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
    cat_names = ['background'] +[cat['name'] for cat in cats]

    # build a map from cat_ids to to the corresponded index of cat_names
    cat_to_label = dict(zip([0] + cat_ids, range(len(cat_names))))

    # get images annotated with the filtered categories
    img_ids = dict((id, len(coco.getImgIds(catIds=[id]))) for id in cat_ids)
    img_ids = [item for sublist in img_ids for item in sublist]  # List of lists flattening
    img_names = [coco.loadImgs(img_id)[0]['file_name'] for img_id in img_ids]

    return img_ids, img_names, cat_names, cat_to_label


def show_im(img_path, img=None):
    """
    Auxiliary function to plot images.

    Parameters
    ----------
    img_path : str
      The file location from which read the image. It is ignored if you pass img.
    img : np.array, optional
      If you have a np.array object in a form of image, you can pass it direcly.
    """

    fig = plt.figsize=(10,)

    # read image from the defined path if img it is available
    if img is None:
        img = mpimg.imread(img_path)

    plt.imshow(img)
    plt.axis('off')
    plt.show()


def show_mask(img, mask, labels, alpha=0.6, figsize=(5,5)):
    """
    Auxiliary function to plot images with mask overlapped.

    Parameters
    ----------
    img : np.array
        Image to plot as background
    mask : np.array
        Mask to plot as foreground
    labels : list
        List of categories containing also the background category.
        Its size must be equal to the number of channels in the mask
    alpha : double, optional
        Transparency of the mask. Default is 0.6
    figsize : pair, optional
        Size of the figure to plot
    """
    plt.figure(figsize=figsize)

    # Preparing the mask with overlapping
    mask_plot = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.float32)
    for i in range(len(labels)):
        mask_plot += mask[:, :, i] * i
        mask_plot[mask_plot >= i] = i
    values = np.array(np.unique(mask_plot), dtype=np.uint8)

    # plot image as background
    plt.imshow(img)

    # plot foreground mask
    im = plt.imshow(mask_plot, interpolation='none', alpha=alpha)

    # legend for mask colors
    colors = [im.cmap(im.norm(value)) for value in range(len(labels))]
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in values]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.axis('off')
    plt.show()


def show_multiple_im(input_img_paths, target_img_paths, cat_names, predictions=None, figsize=(10, 10)):
    """
    Auxiliary function to plot a batch of images.

    Parameters
    ----------
    input_img_paths : str
        The file locations from which read the input images.
    target_img_paths : str
        The file locations from which read the masks.
    cat_names : list
        Array of the categories to show on segmentation masks
    predictions : np.array, optional
        Batch of predictions.
    figsize :  array
        Size of the figure to plot.
    """

    n_images = len(input_img_paths)

    # plot 3 or 2 columns if predictions is available or not
    ncols = 2 if predictions is None else 3

    fig, axes = plt.subplots(figsize=figsize, nrows=n_images, ncols=ncols)

    for i, (input_img_path, target_img_path) in enumerate(zip(input_img_paths, target_img_paths)):
        img = mpimg.imread(input_img_path)
        ax = axes[i, 0] if n_images > 1 else axes[0]
        ax.imshow(img)
        ax.axis('off')

        mask = img_to_array(load_img(target_img_path, color_mode="grayscale"))
        ax = axes[i, 1] if n_images > 1 else axes[1]
        im = ax.imshow(mask[:, :, 0])
        ax.axis('off')

        values = np.array(np.unique(mask), dtype=np.uint8)
        colors = [im.cmap(im.norm(value)) for value in range(len(cat_names))]
        patches = [mpatches.Patch(color=colors[i], label=cat_names[i]) for i in values]
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        if predictions is not None:
            ax = axes[i, 2] if n_images > 1 else axes[2]
            ax.imshow(predictions[i])
            ax.axis('off')

            values = np.array(np.unique(predictions[i]), dtype=np.uint8)
            colors = [im.cmap(im.norm(value)) for value in range(len(cat_names))]
            patches = [mpatches.Patch(color=colors[i], label=cat_names[i]) for i in values]
            ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # set column title
    col_titles = ['Input', 'Target', 'Predictions']
    if n_images > 1:
        for ax, col in zip(axes[0], col_titles):
            ax.set_title(col)
    else:
        for ax, col in zip(axes, col_titles):
            ax.set_title(col)

    fig.tight_layout()
    plt.show()

'''
# TODO REMOVE THIS 

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
    aug_generator_args = dict(rotation_range=10,
                              width_shift_range=0.01,
                              height_shift_range=0.01,
                              brightness_range=(0.8, 1.2),
                              shear_range=0.01,
                              zoom_range=[1, 1.5],
                              horizontal_flip=True,
                              vertical_flip=True,
                              fill_mode='constant',
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
'''

