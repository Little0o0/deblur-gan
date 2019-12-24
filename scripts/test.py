import numpy as np
import cv2
from PIL import Image
import click
import math

from deblurgan.model import generator_model
from deblurgan.utils import load_images, deprocess_image


def get_mse(y_test, x_test):
    return np.mean(np.mean((y_test/255.-x_test/255.) ** 2))

def get_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
        [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def ssim(img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


def get_ssim(y_one, x_one):

    ssims = []
    for i in range(3):
        ssims.append(ssim(y_one[:, :, i], x_one[:, :, i]))
    return np.array(ssims).mean()

def calc_metrics(img1, img2, crop_border, test_Y=True):
    #
    img1 = img1 / 255.
    img2 = img2 / 255.

    if test_Y and img1.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
        im1_in = rgb2ycbcr(img1)
        im2_in = rgb2ycbcr(img2)
    else:
        im1_in = img1
        im2_in = img2

    if im1_in.ndim == 3:
        cropped_im1 = im1_in[crop_border:-crop_border, crop_border:-crop_border, :]
        cropped_im2 = im2_in[crop_border:-crop_border, crop_border:-crop_border, :]
    elif im1_in.ndim == 2:
        cropped_im1 = im1_in[crop_border:-crop_border, crop_border:-crop_border]
        cropped_im2 = im2_in[crop_border:-crop_border, crop_border:-crop_border]
    else:
        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im1_in.ndim))

    psnr = get_psnr(cropped_im1 * 255, cropped_im2 * 255)
    ssim = get_ssim(cropped_im1 * 255, cropped_im2 * 255)
    return psnr, ssim


def test(batch_size):
    data = load_images('./images/test', batch_size)
    y_test, x_test = data['sharp_imgs'], data['blur_imgs']
    print(x_test.shape)
    g = generator_model()
    g.load_weights('generator.h5')
    generated_images = g.predict(x=x_test, batch_size=batch_size)
    generated = np.array([deprocess_image(img) for img in generated_images])
    x_test = deprocess_image(x_test)
    y_test = deprocess_image(y_test)

    psnrs = []
    ssims = []
    for i in range(batch_size):
        psnrs.append(get_psnr(y_test[i, :, :, :], x_test[i, :, :, :]))
        ssims.append(get_ssim(y_test[i, :, :, :], x_test[i, :, :, :]))

    print("MSE is %f" % get_mse(y_test, x_test))
    print("PSNR is %f" % np.mean(psnrs))
    print("SSIM is %f" % np.mean(ssims))


    y = y_test[0, :, :, :]
    x = x_test[0, :, :, :]
    img = generated[0, :, :, :]
    output = np.concatenate((y, x, img), axis=1)
    im = Image.fromarray(output.astype(np.uint8))
    im.save('results{}.png'.format(0))


@click.command()
@click.option('--batch_size', default=4, help='Number of images to process')
def test_command(batch_size):
    return test(batch_size)


if __name__ == "__main__":
    test_command()
