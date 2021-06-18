import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from utils import segmentation_utils
from PIL import Image


class DataGenerator(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, coco, batch_size, img_size, img_ids, img_paths,  cat_to_label, aug_args=None, shuffle=True):
        self.coco = coco
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_paths = img_paths
        self.img_ids = img_ids
        self.cat_to_label = cat_to_label
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """ Number of batches per epoch """
        return len(self.img_ids) // self.batch_size

    def __getitem__(self, idx):
        """ Generate one batch of data """

        # Generate indexes of the batch
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        # n of categories
        n_cat = len(self.cat_to_label)

        # Find list of img paths and image ids to read in the batch
        batch_img_paths = [self.img_paths[i] for i in indexes]
        batch_img_ids = [self.img_ids[i] for i in indexes]

        # load imgs
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype=np.float32)
        for j, path in enumerate(batch_img_paths):
            img = img_to_array(load_img(path, target_size=self.img_size))
            x[j] = img

        # load masks
        y = np.zeros((self.batch_size,) + self.img_size + (n_cat,), dtype=np.float32)
        for j, img_id in enumerate(batch_img_ids):
            mask = segmentation_utils.load_mask(self.coco, img_id, self.cat_to_label)

            # scale down mask
            mask_scaled = np.zeros(self.img_size+ (n_cat,), dtype=np.uint8)
            for i in range(n_cat):
                mask_scaled[:,:,i] = Image.fromarray(mask[:,:,i]).resize(self.img_size, Image.NEAREST)

            y[j] = np.array(mask_scaled, dtype=np.float32)

        if aug_args is not None:
            x, y = segmentation_utils.augment(x, y, aug_args)

        return x, y

    def on_epoch_end(self):
        """ Method called at the end of every epoch that updates indexes """
        self.indexes = np.arange(len(self.img_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


cat_names = ['water'
             , 'bread-white'
             , 'salad-leaf-salad-green'
             , 'tomato'
             , 'butter'
             , 'carrot'
             , 'coffee-with-caffeine'
             , 'rice'
             , 'egg'
             , 'mixed-vegetables'
             , 'wine-red'
             , 'apple'
             , 'jam'
             , 'potatoes-steamed'
             , 'banana'
             , 'cheese' ]

batch_size = 32
img_size = (160,160)
coco = segmentation_utils.load_coco_dataset('../data/val/')

# USE GENERATOR
''' 
img_ids, img_names, cat_names, cat_to_label = segmentation_utils.load_imgs(coco, cat_names)
print(img_ids)
print(img_names)
print(cat_names)
print(cat_to_label)
print("Dataset size: ",len(img_ids))

img_paths = [os.path.join("../data/val/images/", file_name) for file_name in img_names]


data_gen = DataGenerator(coco, batch_size, img_size, img_ids, img_paths,  cat_to_label)

i = random.randint(0,batch_size)
x = data_gen.__getitem__(0)[0]
y = data_gen.__getitem__(0)[1]
print("x shape:",x.shape)
print("y shape:",y.shape)


img = x[i,:,:,:]/255
mask = np.array(y[i,:,:,0], dtype = np.uint8)

show_im("", img)
show_im("", mask)

print(img.shape)
print(mask.shape)

print("The values in the mask are: {}".format(np.unique(y[i,:,:,0])))

'''

img_ids, img_names, cat_names, cat_to_label = segmentation_utils.load_imgs(coco, ['bread-white','jam', 'banana', 'apple'])
class_dist = segmentation_utils.get_class_dist(coco, cat_names)
print(cat_names)
print(cat_to_label)
print(class_dist)
print(segmentation_utils.get_class_weights(class_dist))
# LOAD ALL DATA
x, y = segmentation_utils.load_data(coco, '../data/val/images', img_size, ['apple'])




aug_args = dict(rotation_range=5,
                width_shift_range=0.01,
                height_shift_range=0.01,
              brightness_range=(0.8, 1.2),
              shear_range=0.01,
              zoom_range=[0.7, 1],
              horizontal_flip=True,
              vertical_flip=True,
              fill_mode='constant',
              data_format='channels_last')

x, y = segmentation_utils.augment(x, y, aug_args)
#for i in range(5):
# segmentation_utils.show_mask(x[i,:,:]/255, y[i,:,:,:], cat_names)
