# Sample conversion script from the binary masks ({0, 1}^h×w×q) to the format required by HoVerNet
# Takes as input the base dataset path and the folder name of the dataset to convert. This dataset is expected in the
# format:
# datasets_folder/dataset_name
# |- [image_name]
# |  |- image
# |  |  |- [image_name].png
# |  |- masks
# |  |  |- mask_1.png
# |  |  |- mask_2.png
# |  |  |- [...]
# |- [...]
#
# The output consists in the folders in HoVerNet format, i.e.
# datasets_folder/dataset_name_hnformat
# |- Images
# |  |- image_000.png
# |  |- image_001.png
# |  |- [...]
# |- Masks
# |  |- image_000.mat
# |  |- image_001.mat
# |  |- [...]
# Where mat files contain a 'inst_map' field representing the mask

import os
import numpy as np
from PIL import Image
import time
import shutil
import scipy.io as sio
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description="Sample conversion script")

parser.add_argument("--datasets_folder", default="./datasets/", type=str,
                    help="Path to the folder containing datasets")
parser.add_argument("--dataset_name", default="gen_images", type=str,
                    help="Name of the folder containing the source dataset")

args = parser.parse_args()
DATASET_FOLDER = args.dataset_folder
DATASET_NAME = args.dataset_name

in_folder = os.path.join(DATASET_FOLDER, DATASET_NAME)
out_folder = os.path.join(DATASET_FOLDER, DATASET_NAME + "_hnformat")

# Removes a previous dataset if needed, then creates the new folders
if os.path.exists(out_folder):
    for t in range(0, 4)[::-1]:
        print(f"Deleting previous dataset in {t}     ", end='\r')
        time.sleep(1)
    shutil.rmtree(out_folder)

image_out = os.path.join(out_folder, "Images")
mask_out = os.path.join(out_folder, "Labels")

os.makedirs(image_out, exist_ok=True)
os.makedirs(mask_out, exist_ok=True)

# Converts each image
for n, image_name in enumerate(tqdm(os.listdir(in_folder))):
    # Loads the image
    image_folder_path = os.path.join(in_folder, image_name, 'image')
    image_path = os.path.join(image_folder_path, os.listdir(image_folder_path)[0])
    image_pil = Image.open(image_path).convert('RGB').resize((256, 256))
    image = np.array(image_pil)

    # Loads the masks and converts them to HoVerNet's GT format
    masks_folder_path = os.path.join(in_folder, image_name, 'masks')
    colored_blob = np.zeros((image.shape[0], image.shape[1]))
    for i, blob_file in enumerate(os.listdir(masks_folder_path), start=1):
        blob_path = os.path.join(masks_folder_path, blob_file)
        full_blob = np.array(Image.open(blob_path).convert('L')).astype(bool)
        colored_blob = np.where(full_blob, i, colored_blob)

    # Saves both the image and the new mask
    image_pil.save(os.path.join(image_out, f"image_{n:03d}.png"))
    sio.savemat(os.path.join(mask_out, f"image_{n:03d}.mat"), {"inst_map": colored_blob})

print()
