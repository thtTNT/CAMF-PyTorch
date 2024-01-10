import PIL
import argparse
import os
import numpy as np
from PIL import Image

def window_crop(image, stride, size):
    """
    Crop image into windows of size 'size' with stride 'stride'
    """
    windows = []
    for i in range(0, image.shape[0] - size[0], stride):
        for j in range(0, image.shape[1] - size[1], stride):
            windows.append(image[i : i + size[0], j : j + size[1]])
    return windows


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="input directory of images")
parser.add_argument("--output_dir", required=True, help="output directory of images")

args = parser.parse_args()

# walk directory
for root, dirs, files in os.walk(args.input_dir):
    for file in files:
        if file.endswith(".bmp"):
            image = Image.open(os.path.join(root, file))
            image = np.array(image)
            windows = window_crop(image, 24, (256, 256))
            for i, window in enumerate(windows):
                assert window.shape == (256, 256)
                PIL.Image.fromarray(window).save(
                    os.path.join(args.output_dir, file[:-4] + "_" + str(i) + ".bmp")
                )
