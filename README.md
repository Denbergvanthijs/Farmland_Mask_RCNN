# Mask R-CNN for Object Detection and Segmentation

**NOTE: This repository is a fork of the [original](https://github.com/matterport/Mask_RCNN.git) Mask R-CNN repository by Matterport.**

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

## Getting Started

To train the model run the following in the root of this repository:

```bash
  python ./farmland/farms.py train --dataset=datasets/farmpoly --weights=coco
```

The dataset can be found in [this Google Drive](https://drive.google.com/open?id=1RtAnQD1BTqPCXl_2qsh3m_bYv50fJZ7a) Folder. Place the dataset in the root directory of this repository.

* [inspect_farm_data.ipynb](farmland/inspect_farm_data.ipynb) show the training and validation data. This notebook can be used to see if data gets loaded in properly.

* [inspect_farm_model.ipynb](farmland/inspect_farm_model.ipynb) shows the trained farmland-model. This notebook can be used to see your predictions.

## Training on Your Own Dataset

Start by reading this [blog post about the balloon color splash sample](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46). It covers the process starting from annotating images to training to using the results in a sample application.

In summary, to train the model on your own dataset you'll need to extend two classes:

```Config```
This class contains the default configuration. Subclass it and modify the attributes you need to change.

```Dataset```
This class provides a consistent way to work with any dataset.
It allows you to use new datasets for training without having to change the code of the model. It also supports loading multiple datasets at the same time, which is useful if the objects you want to detect are not all available in one dataset.

## Citation

Use this bibtex to cite the [original](https://github.com/matterport/Mask_RCNN) repository:

```citation
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
```

## Requirements

Python 3.7.6, TensorFlow 1.13.1, Keras 2.0.8 and other common packages listed in `requirements.txt`.

### MS COCO Requirements

To train or test on MS COCO, you'll also need:

* pycocotools (installation instructions below)
* [MS COCO Dataset](http://cocodataset.org/#home)
* Download the 5K [minival](https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0)
  and the 35K [validation-minus-minival](https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0)
  subsets. More details in the original [Faster R-CNN implementation](https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/README.md).

If you use Docker, the code has been verified to work on
[this Docker container](https://hub.docker.com/r/waleedka/modern-deep-learning/).

## Installation

1. Clone this repository
2. Install dependencies

   ```bash
   pip3 install -r requirements.txt
   ```
  
3. Run setup from the repository root directory

    ```bash
    python3 setup.py install
    ```

4. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).
5. (Optional) To train or test on MS COCO install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)
