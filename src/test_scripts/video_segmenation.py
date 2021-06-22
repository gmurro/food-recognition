import cv2
import time
import numpy as np
import PIL
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from PIL import ImageOps
from tensorflow.keras.models import model_from_json

# target img_size
img_size = (128, 128)

cat_names = ['background',
            'water',
            'egg',
            'butter',
            'bread-white',
            'jam',
            'apple',
            'cheese',
            'carrot',
            'salad-leaf-salad-green',
            'banana',
            'mixed-vegetables',
            'tomato',
            'wine-red',
            'rice',
            'coffee-with-caffeine',
            'potatoes-steamed']

# load json and create model
with open('../models/model_D/model_D.json') as file:
    model_json = file.read()
model = model_from_json(model_json)

# load weights
model.load_weights('../models/model_D/weights_model_D_80e.h5')

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

# define a video capture object
vid = cv2.VideoCapture('../../assets/img/video.mp4')

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    camera_img_size = (frame.shape[1], frame.shape[0])

    # run segmentation
    resized_frame = cv2.resize(frame, img_size)
    x = np.expand_dims(resized_frame, axis=0)
    y = model.predict(x)

    mask = np.argmax(y[0], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = ImageOps.autocontrast(tf.keras.preprocessing.image.array_to_img(mask))
    img = tf.keras.preprocessing.image.img_to_array(img.resize(camera_img_size))

    maskImg = np.zeros(frame.shape, frame.dtype)
    maskImg[:, :, 0] = img[:,:,0]
    maskImg[:, :, 1] = img[:,:,0]
    maskImg[:, :, 2] = img[:,:,0]

    ids = np.unique(mask)
    [print(cat_names[int(id)]) if id != 0 and id < 17 else print("") for id in ids]

    overlap = cv2.addWeighted(frame, 1, maskImg, 0.6, 0)

    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()

    # Calculating the fps

    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)

    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)

    # putting the FPS count on the frame
    cv2.putText(overlap, fps, (7, 50), font, 1, (100, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', overlap)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()