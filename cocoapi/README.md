COCO API - http://cocodataset.org/

COCO is a large image dataset designed for object detection, segmentation, person keypoints detection, stuff segmentation, and caption generation. This package provides Matlab, Python, and Lua APIs that assists in loading, parsing, and visualizing the annotations in COCO. Please visit http://cocodataset.org/ for more information on COCO, including for the data, paper, and tutorials. The exact format of the annotations is also described on the COCO website. The Matlab and Python APIs are complete, the Lua API provides only basic functionality.

In addition to this API, please download both the COCO images and annotations in order to run the demos and use the API. Both are available on the project website.
-Please download, unzip, and place the images in: coco/images/
-Please download and place the annotations in: coco/annotations/
For substantially more details on the API please see http://cocodataset.org/#download.

After downloading the images and annotations, run the Matlab, Python, or Lua demos for example usage.

To **install on Windows**, you can use [Anaconda environment](https://www.anaconda.com/products/individual) and you must have the [Visual C++ 2015 build tools](https://go.microsoft.com/fwlink/?LinkId=691126):

1. Clone this repo
2. Create a new environment: `conda create -n deep`
3. Activate the environment: `activate deep`
4. Install requirements:  `pip install -r requirements.txt`
5. inside PythonAPI folder, we can find `setup.py` file. Open the file and delete the following line:
```python
    extra_compile_args=[‘-Wno-cpp’, ‘-Wno-unused-function’, ‘-std=c99’]
```
6. run command `python setup.py build_ext install`
7. run command `python setup.py build_ext --inplace`

