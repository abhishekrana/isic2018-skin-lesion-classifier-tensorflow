##########################################################################################
# Utility functions for images
##########################################################################################
from PIL import Image
import numpy as np
import os
import glob
from matplotlib.image import imread

def load_image(path, size=None):
    """
    Load the image from the given file-path and resize it to the given size if not None.
    Eg: size = (width, height)
    """
    img = Image.open(path)

    if (size != None) and (size != ''):
        img = img.resize(size=size, resample=Image.LANCZOS)

    img = np.array(img)

    # Scale image-pixels so they fall between 0.0 and 1.0
    img = img / 255.0

    # Convert 2-dim gray-scale array to 3-dim RGB array.
    if (len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    return np.array(img)


def load_images(image_paths):
    # Load the images from disk.
    images = [imread(path) for path in image_paths]

    # Convert to a numpy array and return it.
    return np.asarray(images)


def get_images_path_list_from_dir(dir_path, img_format='jpg'):
    img_regex = os.path.join(dir_path, '*.' + img_format)
    img_paths = glob.glob(img_regex)

    # imgs = [load_image(img_path) for img_path in img_paths]
    # return np.array(imgs), img_paths

    return img_paths


def save_image(image_np, image_path_name):
    img = Image.fromarray(image_np)
    img.save(image_path_name)

