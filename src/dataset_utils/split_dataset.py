import os
import math
import time
import shutil

BASE_NAME = "dataset_gen_vet63full_256_correction"

IN_PATH = f"/home/robertob98/datasets/_hovernet_training/{BASE_NAME}" #+ "_hnformat"
OUT_BASE_PATH = f"/home/robertob98/datasets/_hovernet_training/{BASE_NAME}" + "_hnformat_split"

images_in_path = os.path.join(IN_PATH, "Images")
masks_in_path = os.path.join(IN_PATH, "Labels")

datasets = {
    "train": .9,
    "valid": .1,
    # "test": .1
}

files = os.listdir(images_in_path)
n_files = len(files)

if os.path.exists(OUT_BASE_PATH):
    for t in range(0, 4)[::-1]:
        print(f"Deleting previous splits in {t}     ", end='\r')
        time.sleep(1)
    shutil.rmtree(OUT_BASE_PATH)

paths = {}
split_start = 0
split_end = 0
for ds_name, split_perc in datasets.items():
    out_folder_path = os.path.join(OUT_BASE_PATH, ds_name)

    image_out = os.path.join(out_folder_path, "Images")
    mask_out = os.path.join(out_folder_path, "Labels")
    os.makedirs(image_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    split_end = split_start + math.floor(n_files * split_perc)

    file_names = [f.split('.')[0] for f in files[split_start:split_end + 1]]

    for i, file_name in enumerate(file_names, start=1):
        print(f"Splitting {ds_name}: {i}/{len(file_names)}                   ", end='\r')
        image_name = file_name + '.png'
        mask_name = file_name + '.mat'
        image_path = os.path.join(images_in_path, image_name)
        mask_path = os.path.join(masks_in_path, mask_name)
        shutil.copy2(image_path, os.path.join(image_out, image_name))
        shutil.copy2(mask_path, os.path.join(mask_out, mask_name))
    print()

    split_start = split_end + 1
