# Parallelized dataset generation pipeline

import numpy as np
import os
from PIL import Image
import copy
import random
import time
import subprocess
import utils
import shutil
from tqdm import tqdm
import traceback 
import multiprocessing as mp
import argparse
from pathlib import Path
import pickle
import scipy.io as sio
from itertools import chain
import torch
import perlin_torch

def single_interpolation(args):
    """Interpolation of a single blob couple.

    Used for multiprocessing.

    Params:
        args: Tuple containing:
            blobs: the blobs to be interpolated
            zoom: the rescaling factor for perimeter interpolation
            n_points_perimeter: number of points to sample on both perimeters
            n_interp: number of interpolated blobs to generate
    Returns:
        The list of interpolated blobs 
    """
    try:
        blobs, zoom, n_points_perimeter, n_interp = args
        couple = random.sample(blobs, 2)
        return utils.interpolate_blobs(*couple, zoom, n_points_perimeter, n_interp=n_interp, register=True)
    except Exception as e:
        # The exception is printed for multiprocessing
        traceback.print_exc()
        raise e

def single_place_and_transfer(args):
    """Creation of a new ground truth and image.

    Used for multiprocessing.

    Params:
        args: Tuple containing:
            gen_blobs: interpolated blobs
            min_blobs: minimum number of blobs to put in an image
            max_blobs: maximum number of blobs to put in an image
            new_shape: shape of the image
            decoder_path: AdaIN decoder path
            nuclei_metadata: stats computed on the available images
            style_policy: policy for styles. Either 'random', 'closest' or a file name
            mask_cache_path: path to a temporary file (for style transfer)
            output_path: folder for saving the new image
            i: iteration number (to be appended to the image name)
    """
    gen_blobs, min_blobs, max_blobs, new_shape, prior_params, adain_path, decoder_path, nuclei_metadata, style_policy, \
        style_path, mask_cache_base_path, centroids_pdf, centroids_bin_edges, \
        images_path, labels_path, i, plot_path = args
    
    try:
        # If the new shape is not given, it is sampled from those of real images
        if new_shape is None:
            new_shape = random.choice(nuclei_metadata['shapes'])

        # 2. Blob placement
        prior_map = perlin_torch.compute_perlin_image(size=new_shape, seed=random.randint(0, 1000), 
                                                      **prior_params).detach().numpy()

        # Chooses the number of blobs to place
        n_blobs = random.randint(min_blobs, max_blobs)
        # Samples the blobs
        target_blobs = random.sample(gen_blobs, n_blobs)
        
        # Places the blobs greedily
        placed_masks = utils.greedy_placement_spaced(prior_map, target_blobs, centroids_pdf, centroids_bin_edges)
        n_placed = len(placed_masks)

        # If the image is empty, the process terminates
        if n_placed == 0:
            print(f"{i} unable to place")
            return
        
        # Select a style based on the chosen policy
        if style_policy == 'random':
            selected_style_path = os.path.join(style_path, random.choice(os.listdir(style_path)))
        elif style_policy == 'closest':
            selected_style_path = random.choice(utils.closest_style(nuclei_metadata['n_nuclei'], n_placed))
        else:
            selected_style_path = os.path.abspath(style_policy)
        
        mat_mask = np.zeros(placed_masks[0].shape[:2])
        for mask_i, mask in enumerate(placed_masks, start=1):
            mat_mask = np.where(mask, mask_i, mat_mask)

        sio.savemat(os.path.join(labels_path, f"image_{i}.mat"), {"inst_map": mat_mask})
        
        # 3. Style transfer

        # Saves the flattened mask as content
        mask_cache_path = os.path.join(mask_cache_base_path, f"mask_cache_{i}.png")
        Image.fromarray(np.logical_or.reduce(placed_masks, axis=0)).convert('RGB').save(mask_cache_path)
        
        # Construction of the AdaIN call
        adain_command = ['python', 'test.py', '--content', mask_cache_path, '--style', 
                        selected_style_path, '--decoder', decoder_path,
                        '--output', images_path, "--save_ext", f".png", "--i", str(i)]
        
        # Style transfer
        subprocess.run(adain_command, stdout=subprocess.PIPE, cwd=adain_path)

        # Image resizing to the mask dimension
        image_path = os.path.join(images_path, f"image_{i}.png")
        Image.open(image_path).resize(new_shape).save(image_path)

        if plot_path is not None:
            utils.plot_result(prior_map, image_path, placed_masks, os.path.join(plot_path, f"overlay_{i}.png"))
        os.remove(mask_cache_path)
    except Exception as e:
        # The exception is printed for multiprocessing
        traceback.print_exc()
        raise e

if __name__ == '__main__':
    # Multiprocessing setting for Linux-based platforms
    mp.set_start_method('forkserver', force=True)

    # Expert-driven settings

    parser = argparse.ArgumentParser(description="Generation pipeline")

    parser.add_argument("--experiment_name", default="gen_images", type=str,
                        help="Name of the experiments, used for saving")
    parser.add_argument("--n_images", default=500, type=int,
                        help="Number of images to generate")
    parser.add_argument("--new_shape", default=None, nargs=2, type=int,
                        help="Shape of new images to be generated. By default randomly selects the shapes of real "
                             "images")
    parser.add_argument("--base_output_path", default="./gen_dataset", type=str,
                        help="Base output path")
    parser.add_argument("--n_threads", default=32, type=int,
                        help="Number of parallel threads for multiprocessing")
    parser.add_argument("--show_examples", dest="show_examples", action="store_true",
                        help="If passed, saves example images and annotations (may slow down the generation)")
    parser.add_argument("--discard_previous", dest="discard_previous", action="store_true",
                        help="If passed, everything from previous experiments with the same is discarded")
    
    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed for experiments")

    interp_group = parser.add_argument_group("Interpolation parameters")
    interp_group.add_argument("--real_path", default="./real_images/", type=str,
                              help="Path to input blobs")
    interp_group.add_argument("--n_couples", default=500, type=int,
                              help="Number of couples to interpolate")
    interp_group.add_argument("--n_interp", default=30, type=int,
                              help="Number of interpolated blobs for each couple")
    interp_group.add_argument("--resolution_mult", default=10, type=int,
                              help="Resolution multiplier for initial blobs")
    interp_group.add_argument("--n_points_perimeter", default=500, type=int,
                              help="Number of points to sample on each perimeter")
    interp_group.add_argument("--blob_cache_file", default="interpolation/blobs_cache.pickle", type=str,
                              help="Cache file for nuclei blobs (in output_path, created if not found)")
    
    placement_group = parser.add_argument_group("Placement parameters")
    placement_group.add_argument("--min_blobs", default=None, type=int,
                                 help="Minimum number of blobs to add to an image. By default, selects the minimum "
                                      "number according to real images")
    placement_group.add_argument("--max_blobs", default=None, type=int,
                                 help="Maximum number of blobs to add to an image. By default, selects the maximum "
                                      "number according to real images")
    placement_group.add_argument("--centroid_dist_file", default="placement/centroid_distances.json", type=str,
                                 help="File containing the distance distribution of nuclei centroids. Created from data"
                                      " if not found")
    placement_group.add_argument("--prior_file", default="placement/prior.json", type=str,
                                 help="File containing the Perlin prior parameters. Created from data if not found")
    
    transfer_group = parser.add_argument_group("Style transfer parameters")
    transfer_group.add_argument("--adain_path", default="./pytorch-AdaIN/", type=str,
                                help="Path to AdaIN's folder")
    transfer_group.add_argument("--adain_decoder_path", default="./style_transfer/decoder_iter_160000.pth.tar", 
                                type=str, help="Path to AdaIN's decoder")
    transfer_group.add_argument("--styles_path", default="./style_transfer/styles", type=str,
                                help="Path to the folder containing styles for AdaIN")
    transfer_group.add_argument("--style_policy", default="closest", type=str,
                                help="Policy for the style. If 'random' a random style from those available is selected"
                                     " each time. If 'closest' the style with the closest number of nuclei is selected."
                                     " If anything else, represents the name of the style file to use for each image")
    transfer_group.add_argument("--metadata_file", default="style_transfer/nuclei_metadata.json", type=str,
                                help="File containing information about real images (in output_path, will"
                                " be created if not found")
    transfer_group.add_argument("--mask_cache_folder", default="mask_cache/", type=str,
                                help="Folder used to store temporary masks for style transfer (in output_path, will be "
                                     "deleted)")
    
    # Parsing
    args = parser.parse_args()

    BASE_NAME = args.experiment_name
    N_IMAGES = args.n_images
    NEW_SHAPE = args.new_shape
    BASE_OUTPUT_PATH = args.base_output_path
    N_THREADS = args.n_threads
    SHOW_EXAMPLES = args.show_examples
    SEED = args.seed
    DISCARD_PREVIOUS = args.discard_previous

    REAL_PATH = args.real_path
    N_COUPLES = args.n_couples
    N_INTERP = args.n_interp
    ZOOM = args.resolution_mult
    N_POINTS_PERIMETER = args.n_points_perimeter
    BLOB_CACHE_FILE = args.blob_cache_file

    MIN_BLOBS = args.min_blobs
    MAX_BLOBS = args.max_blobs
    CENTROID_DIST_FILE = args.centroid_dist_file
    PRIOR_FILE = args.prior_file

    ADAIN_PATH = args.adain_path
    DECODER_PATH = args.adain_decoder_path
    STYLES_PATH = args.styles_path
    STYLE_POLICY = args.style_policy
    METADATA_FILE = args.metadata_file
    MASK_CACHE_FOLDER = args.mask_cache_folder

    # Initialization
    random.seed(SEED)
    np.random.seed(SEED)

    # Warning before removing a previous experiment with the same name
    output_path = os.path.join(BASE_OUTPUT_PATH, BASE_NAME)
    if DISCARD_PREVIOUS:
        if os.path.exists(output_path):
            for t in range(0, 4)[::-1]:
                print(f"Discarding previous EXPERIMENT in {t}     ", end='\r')
                time.sleep(1)
            shutil.rmtree(output_path)

    # Output paths creation
    os.makedirs(output_path, exist_ok=True)
    print(f"Saving output to {os.path.abspath(output_path)}")

    examples_path = os.path.join(output_path, "examples")
    plot_path = os.path.join(examples_path, "plots")

    gen_dataset_path = os.path.join(output_path, f"dataset_{BASE_NAME}")
    if os.path.exists(gen_dataset_path):
        for t in range(0, 4)[::-1]:
            print(f"Deleting previous dataset in {t}     ", end='\r')
            time.sleep(1)
        shutil.rmtree(gen_dataset_path) 
    if plot_path is not None and os.path.exists(plot_path):
        shutil.rmtree(plot_path) 

    print()

    if SHOW_EXAMPLES:
        os.makedirs(examples_path, exist_ok=True)
        os.makedirs(plot_path, exist_ok=True)
    else:
        plot_path = None    

    metadata_path = Path(os.path.join(output_path, METADATA_FILE))
    os.makedirs(metadata_path.parent, exist_ok=True)
    blob_cache_path = Path(os.path.join(output_path, BLOB_CACHE_FILE))
    os.makedirs(blob_cache_path.parent, exist_ok=True)
    mask_cache_path = Path(os.path.join(output_path, MASK_CACHE_FOLDER))
    os.makedirs(mask_cache_path, exist_ok=True)

    centroids_dist_path = Path(os.path.join(output_path, CENTROID_DIST_FILE))
    os.makedirs(centroids_dist_path.parent, exist_ok=True)
    prior_params_path = Path(os.path.join(output_path, PRIOR_FILE))
    os.makedirs(prior_params_path.parent, exist_ok=True)

    images_path = os.path.join(gen_dataset_path, "Images")
    os.makedirs(images_path)
    labels_path = os.path.join(gen_dataset_path, "Labels")
    os.makedirs(labels_path)


    # Loading all blobs and computing the metadata
    blob_path = os.path.join(REAL_PATH, "Labels")
    blobs = utils.load_blobs_mat(blob_path)
    print(f"Loaded {len(blobs)} real blobs from {blob_path}")
    nuclei_metadata = utils.get_nuclei_metadata(REAL_PATH, metadata_path, overwrite=DISCARD_PREVIOUS)

    # 1. Blob generation
    if SHOW_EXAMPLES:
        print("Saving interpolation examples...", end=' ')
        couple = random.sample(blobs, 2)
        gen_blobs, lin_spaces = utils.interpolate_blobs(*couple, ZOOM, N_POINTS_PERIMETER, N_INTERP, 
                                                        return_lin_spaces=True)
        utils.plot_interpolated_blobs(gen_blobs, out_path=os.path.join(examples_path, "interpolated_blobs.png"))
        utils.save_interp_animation(lin_spaces, out_path=os.path.join(examples_path, "interpolation.gif"))
        print("Done")

    if os.path.exists(blob_cache_path):
        # If blobs were already interpolated and cached, they are just reloaded
        with open(blob_cache_path, 'rb') as f:
            gen_blobs = pickle.load(f)
        print("Loaded", end=' ')
    else:
        # Otherwise, we parallelize their generation
        print(f"Generating new blobs ({N_THREADS} threads):")
        args = (blobs, ZOOM, N_POINTS_PERIMETER, N_INTERP)
        all_args = [copy.deepcopy(args) for _ in range(N_COUPLES)]
        with mp.Pool(N_THREADS) as p:
            thread_result = list(tqdm(p.imap(single_interpolation, all_args), total=N_COUPLES))
        
        gen_blobs = []
        for sub_list in thread_result:
            gen_blobs.extend(sub_list)

        with open(blob_cache_path, 'wb') as f:
            pickle.dump(gen_blobs, f)
        print("Created", end=' ')

    print(f"{len(gen_blobs)} masks. Expected: {N_COUPLES * N_INTERP}")

    # 2 & 3. Blob placement and style transfer

    # Distance components
    centroids_pdf, centroids_bin_edges = utils.compute_centroids(centroids_dist_path, blob_path)

    # Prior components
    prior_params = utils.get_prior_parameters(prior_params_path, os.path.join(blob_path, os.listdir(blob_path)[0]), 4)
    prior_params = {k: torch.from_numpy(np.array(v)) for k, v in prior_params.items() if k != "n_octaves"}

    # If parameters are not specified, random ones area sampled from real images
    min_blobs = MIN_BLOBS if MIN_BLOBS is not None else min([int(b) for b in list(nuclei_metadata['n_nuclei'].keys())])
    max_blobs = MAX_BLOBS if MAX_BLOBS is not None else max([int(b) for b in list(nuclei_metadata['n_nuclei'].keys())])

    print(f"Generating new GTs and images ({N_THREADS} threads):")
    args = [(gen_blobs, min_blobs, max_blobs, NEW_SHAPE, prior_params, os.path.abspath(ADAIN_PATH), DECODER_PATH, 
             nuclei_metadata, STYLE_POLICY, STYLES_PATH, mask_cache_path, centroids_pdf, centroids_bin_edges,
             images_path, labels_path, i, plot_path) for i in range(N_IMAGES)]
    with mp.Pool(N_THREADS) as p:
        results = list(tqdm(p.imap(single_place_and_transfer, args), total=N_IMAGES))

    print()
