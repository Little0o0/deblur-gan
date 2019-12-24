import os
from PIL import Image
import numpy as np
import tensorflow as tf


RESHAPE = (256, 256)


def is_an_image_file(filename):
    # Judge whether it is an image file
    IMAGE_EXTENSIONS = ['.png', '.jpeg', '.jpg', '.PNG', 'JPG', 'JPEG']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False


def list_image_files(directory):
    #  show and sort all image files in such directory
    files = sorted(os.listdir(directory))
    return [os.path.join(directory, f) for f in files if is_an_image_file(f)]


def preprocess_image(cv_img):
    # parm cv_img:raw img with differenet size
    # retrun img: a array with size(256,256) and in range(0,1)
    cv_img = cv_img.resize(RESHAPE)
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img


def deprocess_image(img):
    # img in range(0,1) --> in range(0,255)
    img = img * 127.5 + 127.5
    return img.astype('uint8')


def save_image(np_arr, path):
    img = np_arr * 127.5 + 127.5
    im = Image.fromarray(img)
    im.save(path)


def load_images(path, max_images):
    # use for load the images
    blur_paths = list_image_files(os.path.join(path, 'blur'))
    sharp_paths = list_image_files(os.path.join(path, 'sharp'))
    blur_imgs, sharp_imgs = [], []
    blur_imgs_paths, sharp_imgs_paths = [], []
    for b, s in zip(blur_paths, sharp_paths):
        blur_imgs.append(preprocess_image(Image.open(b)))
        sharp_imgs.append(preprocess_image(Image.open(s)))
        blur_imgs_paths.append(b)
        sharp_imgs_paths.append(s)
        if len(blur_imgs) > max_images - 1:
            break

    return {
        'blur_imgs': np.array(blur_imgs), 
        'blur_paths': np.array(blur_imgs_paths), 
        'sharp_imgs': np.array(sharp_imgs),
        'sharp_paths': np.array(sharp_imgs_paths)
    }

# def write_log(callback, names, logs, batch_no):
#     """
#     Util to write callback for Keras training
#     """
#     for name, value in zip(names, logs):
#         summary = tf.Summary()
#         summary_value = summary.value.add()
#         summary_value.simple_value = value
#         summary_value.tag = name
#         callback.writer.add_summary(summary, batch_no)
#         callback.writer.flush()
