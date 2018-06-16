##########################################################################################
# Utility functions for images
##########################################################################################
import os
import cv2
import glob
import numpy as np
from PIL import Image
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
    # img = img / 255.0

    # Convert 2-dim gray-scale array to 3-dim RGB array.
    if (len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    return np.array(img)


def load_images(image_paths):
    # Load the images from disk.
    images = [imread(path) for path in image_paths]

    # Convert to a numpy array and return it.
    return np.asarray(images)

def load_images_v2(image_paths_list):
    images = []
    for image_path in image_paths_list:
        image = Image.open(image_path)
        image = np.array(image, dtype=np.float32)
        images.append(image)
    images = np.array(images)

    # images = np.array([np.array(Image.open(image_path), dtype=np.float32) for image_path in image_paths_list])

    return images


def get_images_path_list_from_dir(dir_path, img_format='jpg'):
    img_regex = os.path.join(dir_path, '*.' + img_format)
    img_paths = glob.glob(img_regex)

    # imgs = [load_image(img_path) for img_path in img_paths]
    # return np.array(imgs), img_paths

    return img_paths


def save_image(image_np, image_path_name):
    img = Image.fromarray(image_np)
    img.save(image_path_name)



def color_constancy(img, power=6, gamma=None):
    """
    Parameters
    ----------
    img: 2D numpy array
        The original image with format of (h, w, c)
    power: int
        The degree of norm, 6 is used in reference paper
    gamma: float
        The value of gamma correction, 2.2 is used in reference paper
    """
    ## img = cv2.imread(img_name)

    # img = np.array(Image.open(img_name))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype('uint8')
        look_up_table = np.ones((256,1), dtype='uint8') * 0
        for i in range(256):
            look_up_table[i][0] = 255*pow(i/255, 1/gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype('float32')
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0,1)), 1/power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec/rgb_norm
    rgb_vec = 1/(rgb_vec*np.sqrt(3))
    img = np.multiply(img, rgb_vec)

    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

    # img_name_mod = img_name.split('.')[0] + '_ilu.' + img_name.split('.')[1]
    # img_save = Image.fromarray(img.astype('uint8'))
    # img_save.save(img_name_mod)

    ## cv2.imwrite(img_name_mod, np.array(img_save))

    # return img
    return img.astype(img_dtype)


