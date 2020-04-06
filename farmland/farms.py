"""
Mask R-CNN.
Train on custom Farmland dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Additions by Thijs van den Berg

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python ./farmland/farms.py train --dataset=datasets/farmpoly --weights=coco

    # Resume training a model that you had trained earlier
    python ./farmland/farms.py train --dataset=datasets/farmpoly --weights=last

    # Apply color splash to an image
    python ./farmland/farms.py splash --weights=datasets/mask_rcnn_farm_bb.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python ./farmland/farms.py splash --weights=last --video=<URL or path to file>
"""

import datetime
import json
import os

import numpy as np
import skimage.draw

from mrcnn import model as modellib
from mrcnn import utils
from mrcnn.config import Config

# Path to trained weights file
COCO_WEIGHTS_PATH = "./datasets/mask_rcnn_farm_bb.h5"  # Pretrained COCO model with trained bounding boxes

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = "./logs"

############################################################
#  Configurations
############################################################


class FarmsConfig(Config):
    """Configuration for training on the farmland dataset
    Derives from the base Config class and overrides some values.
    """

    # Give the configuration a recognizable name
    NAME = "farm"

    # We use Google Colab to Train, this could use two images per GPU.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + farmland classifier

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 60% confidence
    DETECTION_MIN_CONFIDENCE = 0.60
    DETECTION_MAX_INSTANCES = 50

    # CUSTOM, dataset contains only 256x256 images
    IMAGE_RESIZE_MODE = "none"

    # Possibly redundant
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # TODO: improve minimask size

############################################################
#  Dataset
############################################################


class FarmsDataset(utils.Dataset):
    """Loads the Farmland dataset."""

    def load_farms(self, dataset_dir, subset):
        """Loads the pictures and adds them to the class."""
        self.add_class("farmland", 1, "farmland")
        # Source == class. Else, it'll give errors in `inspect_farm_data`

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        json_file = "farmpoly_" + subset + ".json"  # Train of validation subset

        annotations = json.load(open(os.path.join(dataset_dir, json_file)))

        for image in annotations:
            polygons = []
            for poly in image["Label"]["objects"]:
                all_x = [x["x"] for x in poly["polygon"]]  # All x-coords of the polygon
                all_y = [x["y"] for x in poly["polygon"]]  # All y-coords of the polygon

                polygons.append({"all_points_x": all_x, "all_points_y": all_y,
                                 'name': 'polygon'})  # Append every polygon to the list of polys of the image

            image_path = os.path.join(dataset_dir, image["External ID"])
            im_data = skimage.io.imread(image_path)
            im_data = im_data[:, :, :3]  # Cut of the transparent layer containing [255,...,255]
            height, width = im_data.shape[:2]

            # Polygons is a list of dicts, each dict is a polygon with two keys
            # "all_x" containing all x values, [-1] == [0]
            # "all_y" containing all y values, [-1] == [0]
            self.add_image(
                "farmland",
                image_id=image["External ID"],  # use file name as a unique imageid
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Loads the mask of the image."""
        image_info = self.image_info[image_id]
        if image_info["source"] != "farmland":  # Possibly redundant
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "farmland":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    print(os.getcwd())
    dataset_train = FarmsDataset()
    dataset_train.load_farms(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FarmsDataset()
    dataset_val.load_farms(args.dataset, "train")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    # python ./farmland/farms.py train --dataset=datasets/farmpoly --weights=coco

    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect farmland.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/farmland/dataset/",
                        help='Directory of the Farmland dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = FarmsConfig()
    else:
        class InferenceConfig(FarmsConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
