from scripts.test import get_psnr, ssim
from PIL import Image
import numpy as np 

RESHAPE = (256, 256)
x_sharp = np.array(Image.open("output/results.png").resize(RESHAPE))
y_sharp = np.array(Image.open("test/0.png").resize(RESHAPE))

val_ssim = ssim(x_sharp, y_sharp)
val_psnr = get_psnr(x_sharp, y_sharp)

print(val_ssim)
print(val_psnr)