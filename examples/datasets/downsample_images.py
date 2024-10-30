import os
import argparse
from PIL import Image

def downsample_images(input_folder, output_folder, scale_factor):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            new_size = (int(img.width / scale_factor), int(img.height / scale_factor))
            img_resized = img.resize(new_size, Image.LANCZOS)
            img_resized.save(os.path.join(output_folder, filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downsample images in a folder.')
    parser.add_argument('input_folder', type=str, help='Path to the input folder containing images.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder to save downsampled images.')
    parser.add_argument('scale_factor', type=float, help='Scale factor to downsample images.')

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    scale_factor = args.scale_factor
    downsample_images(input_folder, output_folder, scale_factor)