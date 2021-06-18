# :pizza: Food Recognition :poultry_leg:

This repository contains a project realized as part of the *Deep Learning* exam of the [Master's degree in Artificial Intelligence, University of Bologna](https://corsi.unibo.it/2cycle/artificial-intelligence).

The *Food Recognition* challenge is a competition hosted by [AIcrowd](https://www.aicrowd.com/challenges/food-recognition-challenge), where participants should deal with an image segmentation problem aiming to recognize individual food items in each image.

The **goal** of this challenge is to train models which can look at images of food items and detect the individual food items present in them.

## Prerequisites
It is needed to install *pycocotools* to execute the notebooks.
You can get it from `pip` or from the original [repo](https://github.com/cocodataset/cocoapi) 
To **install on Windows**, you can use [Anaconda environment](https://www.anaconda.com/products/individual) and you must have the [Visual C++ 2015 build tools](https://go.microsoft.com/fwlink/?LinkId=691126):
1. Clone the repo
2. Create a new environment: `conda create -n deep`
3. Activate the environment: `activate deep`
4. Install requirements:  `pip install -r requirements.txt`
5. inside PythonAPI folder, we can find `setup.py` file. Open the file and delete the following line:
    ```python
    extra_compile_args=[‘-Wno-cpp’, ‘-Wno-unused-function’, ‘-std=c99’]
    ```
6. run command `python setup.py build_ext install`
7. run command `python setup.py build_ext --inplace`

## Dataset

The dataset for the [AIcrowd Food Recognition Challenge](https://www.aicrowd.com/challenges/food-recognition-challenge) is available at https://www.aicrowd.com/challenges/food-recognition-challenge/dataset_files

This dataset contains :

- train-v0.4.tar.gz : This is the Training Set of 24120 (as RGB images) food images, along with their corresponding 39328 annotations in [MS-COCO format](http://cocodataset.org/#home)
- val-v0.4.tar.gz: This is the suggested Validation Set of 1269 (as RGB images) food images, along with their corresponding 2053 annotations in [MS-COCO format](http://cocodataset.org/#home)
- test_images-v0.4.tar.gz : This is the debug Test Set for Round-3, where you are provided the same images as the validation set.



## Evaluation Criteria

For a known ground truth mask A, you propose a mask B, then we first compute IoU (Intersection Over Union).

IoU measures the overall overlap between the true region and the proposed region. Then we consider it a True detection, when there is at least half an overlap, or when IoU > 0.5

Then we can define the following parameters :

Precision (IoU > 0.5) :Recall (IoU > 0.5)

The final scoring parameters AP*{IoU > 0.5} and AR*{IoU > 0.5} are computed by averaging over all the precision and recall values for all known annotations in the ground truth.

A further discussion about the evaluation metric can be found [here](https://discourse.aicrowd.com/t/evaluation-criteria/2668).



## Results

Final results will be presented as soos as possible.



## Resources & Libraries

* pycocotools [API](https://github.com/cocodataset/cocoapi)
* Tensorflow + Keras



## Versioning

We use Git for versioning.



## Group members

| Reg No. |   Name    |  Surname  |                 Email                  |                       Username                        |
| :-----: | :-------: | :-------: | :------------------------------------: | :---------------------------------------------------: |
| 997317  | Giuseppe  |   Murro   |    `giuseppe.murro@studio.unibo.it`    |         [_gmurro_](https://github.com/gmurro)         |
| 985203  | Salvatore | Pisciotta | `salvatore.pisciotta2@studio.unibo.it` | [_SalvoPisciotta_](https://github.com/SalvoPisciotta) |



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details