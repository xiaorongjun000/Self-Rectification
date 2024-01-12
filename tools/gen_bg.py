from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', type=str, required=True, help='input reference texture path')

args = parser.parse_args()
path = args.path

image = Image.open(path)
resized_image = image.resize((512, 512))
image_array = np.array(resized_image)
height, width, channels = image_array.shape

flattened_pixels = image_array.reshape(-1, channels)
np.random.shuffle(flattened_pixels)
shuffled_image_array = flattened_pixels.reshape(height, width, channels)
shuffled_image = Image.fromarray(shuffled_image_array.astype(np.uint8))

shuffled_image.save('target-bg.jpg')
