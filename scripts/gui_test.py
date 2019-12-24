import numpy as np
from PIL import Image
from deblurgan.model import generator_model
from deblurgan.utils import preprocess_image, deprocess_image

def deblur(inputFile="./input/0.png"):
    #for example imagePath = "E:/deblur-gan/input/0.png"

    #output is in "./output/results{}.png"
    data = np.array( [preprocess_image(Image.open(inputFile)) ])
    g = generator_model()
    g.load_weights('generator.h5')
    generated_images = g.predict(x=data, batch_size=1)
    generated = np.array([deprocess_image(img) for img in generated_images])
    output = generated[0, :, :, :]
    im = Image.fromarray(output.astype(np.uint8))
    im.save('./output/results{}.png'.format(0))


if __name__ == "__main__":
    deblur()
