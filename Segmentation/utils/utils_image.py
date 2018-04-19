##########################################################################################
# Utility functions for images
##########################################################################################
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import glob
import pdb
from matplotlib.image import imread

def load_image(path, size=None):
    """
    Load the image from the given file-path and resize it to the given size if not None.
    Eg: size = (width, height)
    """
    img = Image.open(path)

    # TODO: Check
    if img.format == 'PNG':
        img = img.convert('RGB')

    if (size != None) and (size != ''):
        img = img.resize(size=size, resample=Image.LANCZOS)

    img = np.array(img)

    # TODO: Why needed?
    # Scale image-pixels so they fall between 0.0 and 1.0
    # img = img / 255.0

    # Convert 2-dim gray-scale array to 3-dim RGB array.
    # if (len(img.shape) == 2):
    #     img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    return img


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
    # image_np = np.rollaxis(image_np, 2, 0)
    img = Image.fromarray(image_np, mode='RGB')
    img.save(image_path_name)


def generate_rand_image(img_shape):
    """
    img_shape = (100, 100, 3)
    """
    img_np = np.random.rand(img_shape) * 255

    img = Image.fromarray(img_np.astype('uint8')).convert('RGBA')
    img.save('temp/img_generated.png')

    return img



def tf_image_save(path):
    # tf.image.decode_jpeg returns a tf.uint8 tensor
    img = tf.image.decode_jpeg(tf.read_file(path), channels=3)

    # tf.image.resize_image_with_crop_or_pad returns tf.uint8
    # tf.image.resize_images returns a tf.float32 tensor
    img = tf.cast(tf.image.resize_images(img, [200, 200]), tf.uint8)

    # sess.run(img)
    return img





