# Utilities for the dataset generation
import open3d as o3d
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import cm, pyplot as plt
from PIL import Image
import copy
import scipy
import skimage
from scipy.ndimage import gaussian_filter
from scipy import signal
import random
from time import time
from skimage import morphology
from skimage import segmentation
from scipy.ndimage.morphology import binary_dilation
import cv2
import math
from perlin_noise import PerlinNoise
from colorsys import hsv_to_rgb, rgb_to_hsv
import scipy.io as sio
from scipy.ndimage import center_of_mass
from pathlib import Path
from glob import glob
from typing import Union, List
import pickle
import tqdm
import json
from celluloid import Camera
import noise


#################
# I/O functions #
#################

def load_blobs(blobs_path):
    blobs = []
    for blob_folder in os.listdir(blobs_path):
        image_path = os.path.join(blobs_path, blob_folder)
        for blob_file in os.listdir(image_path):
            if blob_file[0] == '_':
                continue
            file_path = os.path.join(image_path, blob_file)
            full_blob = np.array(Image.open(file_path).convert('L')).astype(bool)
            if np.count_nonzero(full_blob) < 50:
                    continue  
            blob_where = np.argwhere(full_blob)
            min_blob_y = np.min(blob_where[:, 0])
            min_blob_x = np.min(blob_where[:, 1])
            max_blob_y = np.max(blob_where[:, 0])
            max_blob_x = np.max(blob_where[:, 1])

            blob = full_blob[min_blob_y:max_blob_y+1, min_blob_x:max_blob_x+1]
            
            blobs.append(blob)
    return blobs

def load_full_blobs(blobs_path):
    blobs = []
    # for blob_folder in os.listdir(blobs_path):
    image_path = blobs_path
        # image_path = os.path.join(blobs_path, blob_folder)
    for blob_file in os.listdir(image_path):
        if blob_file[0] == '_':
            continue
        file_path = os.path.join(image_path, blob_file)
        full_blob = np.array(Image.open(file_path).convert('L')).astype(bool)
        if np.count_nonzero(full_blob) < 50:
                    continue  
        blobs.append(full_blob)
    return blobs

def load_full_blobs_mat_by_image(blobs_path):
    blobs = []
    for blob_file in os.listdir(blobs_path):
        blob_path = os.path.join(blobs_path, blob_file)
        mask = sio.loadmat(blob_path)['inst_map']
        blobs_in_image = []
        for i in range(1, np.max(mask).astype(int) + 1):
            full_blob = np.where(mask == i, 1, 0).astype(bool)  
            if np.count_nonzero(full_blob) < 50:
                    continue          
            blobs_in_image.append(full_blob)
        blobs.append(np.array(blobs_in_image))
    return blobs

def load_blobs_mat(blobs_path):
    """Loads blobs in HoVerNet format.

    Args:
        blobs_path: path to the 'Labels' folder containing .mat nuclei annotations
    Returns:
        A list of binary blobs
    """
    blobs = []
    for blob_file in os.listdir(blobs_path):
        try:
            blob_path = os.path.join(blobs_path, blob_file)
            mask = sio.loadmat(blob_path)['inst_map']
            for i in range(1, np.max(mask).astype(int) + 1):                
                full_blob = np.where(mask == i, 1, 0).astype(bool)
                if np.count_nonzero(full_blob) < 50:
                    continue
                blob_where = np.argwhere(full_blob)
                min_blob_y = np.min(blob_where[:, 0])
                min_blob_x = np.min(blob_where[:, 1])
                max_blob_y = np.max(blob_where[:, 0])
                max_blob_x = np.max(blob_where[:, 1])
                blob = full_blob[min_blob_y:max_blob_y+1, min_blob_x:max_blob_x+1]
                blobs.append(blob)
        except Exception as e:
            print(e)
    return blobs


def closest_style(nuclei_metadata, target_key):
    closest_key = min(nuclei_metadata.keys(), key=lambda x: abs(int(x) - target_key))
    return nuclei_metadata[closest_key]

def get_nuclei_metadata(real_path, metadata_path, overwrite=True):
    """Computes and returns the metadata of real images

    Args:
        real_path: Path to the folder containing real images and their annotations
        metadata_path: Path to the metadata file
        overwrite: If True will compute the metadata instead of loading an existing file
    Returns:
        A dict 
        {
            'shapes': list of real image shapes,
            'n_nuclei': {
                n nuclei in image: list of paths of images having n nuclei
            }
        }
    """
    if not overwrite and os.path.exists(metadata_path):
        print(f"Loading metadata file: {os.path.abspath(metadata_path)}")
        with open(metadata_path, "r") as f:
            nuclei_metadata = json.load(f)
    else:
        print(f"Creating metadata file {os.path.abspath(metadata_path)}")
        nuclei_metadata = {}
        labels_path = os.path.join(real_path, "Labels")
        images_path = os.path.join(real_path, "Images")
        nuclei_metadata['shapes'] = []
        nuclei_metadata['n_nuclei'] = {}
        for file_name in os.listdir(labels_path):
            nuclei_path = os.path.join(labels_path, file_name)
            style_path = os.path.join(images_path, file_name.split('.')[0] + ".png")
            nuclei = sio.loadmat(nuclei_path)['inst_map']
            nuclei_metadata['shapes'].append(nuclei.shape)
            n_nuclei = int(np.max(nuclei))
            if n_nuclei in nuclei_metadata:
                nuclei_metadata['n_nuclei'][n_nuclei].append(style_path)
            else:
                nuclei_metadata['n_nuclei'][n_nuclei] = [style_path]
        with open(metadata_path, "w") as f:
            json.dump(nuclei_metadata, f)

    return nuclei_metadata

###########################
# Interpolation functions #
###########################

def add_points(src, dst, image):
    """Adds to image the points between src and dst
    
    """
    # Define the gap to cover
    gap = dst - src
    # Initialize the current element
    curr = np.copy(src)
    # As long as we don't reach the destination we move in the space
    while(np.any(gap != 0)):
        # Moves over y
        if gap[0] > 0:
            curr[0] += 1
        elif gap[0] < 0:
            curr[0] -= 1
        # Moves over x
        if gap[1] > 0:
            curr[1] += 1
        elif gap[1] < 0:
            curr[1] -= 1

        # Adds the current point
        image[curr[0], curr[1]] = 1

        # Computes the new gap
        gap = dst - curr
    return image


def get_super_resolution(blob, zoom):
    """Returns a higher-resolution version of the input blob

    Args:
        blob: Input to resize
        zoom: Resolution multiplier
    Returns:
        The resized input blob
    """
    return cv2.resize(blob.astype(np.uint8), (zoom * np.asarray(blob.shape[::-1]))) > 0.5

def get_perimeter(blob):
    """Computes the pixels belonging to the boundary

    Args:
        blob: binary blob
    Returns:
        Binary mask of the blob's perimeter
    """
    # Padding for correct computation
    padded_sample = np.pad(blob, [[1, 1], [1, 1]])
    return (padded_sample ^ morphology.erosion(padded_sample))[1:-1, 1:-1]


def compute_centroid(sample):
    awhere = np.argwhere(sample)
    return np.round(np.mean(awhere[:, 0])).astype(int), np.round(np.mean(awhere[:, 1])).astype(int)


def search_perimeter_neighborhood(perimeter, working_point, max_time=10):
    """Local search in the perimeter.

    Searches for the next point to process near working_point
    
    """
    # Search scope
    scope = 1
    start = time()
    while(scope < np.count_nonzero(perimeter) and time() - start < max_time):
        for i in range(-scope, scope + 1)[::-1]:
            for j in range(-scope, scope + 1)[::-1]:
                # Checks if the analyzed point is still in the image and, if it is part of the perimeter, returns it
                if working_point[0] + i in range(0, perimeter.shape[0]) \
                        and working_point[1] + j in range(0, perimeter.shape[1]) \
                        and perimeter[working_point[0] + i, working_point[1] + j]:
                    return working_point[0] + i, working_point[1] + j
        # If no neighbor was found, enlarges the search scope
        scope += 1

    # If no neighbors are available, returns None
    return None

def radialize(perimeter):
    """Computation of the radial perimeter.

    Starting from an arbitrary point, the function returns the perimeter ordered according to the angle

    Args:
        perimeter: The perimeter to be radialized
    Returns:
        The reordered perimeter
    """
    working_perimeter = np.copy(perimeter)
    start_y = 0
    start_x = np.where(perimeter[start_y])[0][0]

    radialized = []
    i = 0
    
    # Starts from an arbitrary point
    working_point = start_y, start_x
    # For each point in the perimeter
    while(working_point is not None and i <= np.count_nonzero(perimeter)):
        # Marks the point as analyzed
        working_perimeter[working_point] = 0
        # Adds the point to the ordered perimeter
        radialized.append(working_point)
        # Counts the processed points
        i += 1
        # Performs a local search to find the next point
        next_point = search_perimeter_neighborhood(working_perimeter, working_point)
        # Passes to the next point
        working_point = next_point
   
    return np.array(radialized)

def choose_points(radialized, n_points):
    """Selects n_points equally spaced points on the perimeter.

    Args:
        radialized: radialized perimeter
        n_points: number of points to sample

    Returns:
        An array of perimeter points' coordinates
    """
    # Starts from a random point on the perimeter
    start_i = np.random.choice(range(len(radialized)))
    # Computes the interval to have equidistant points
    interval = len(radialized) / n_points
    # Moves the radial perimeter to start from the selected starting point
    rolled_rad = radialized #np.roll(radialized, start_i, axis=0)
    # Samples a point at each interval
    points = []
    for i in np.arange(0, len(radialized) - 1, interval):
        points.append(rolled_rad[i.astype(int)])
        
    return np.array(points)

def interpolate_blobs(blob_1, blob_2, zoom, n_points_perimeter, n_interp, register=True, return_lin_spaces=False):
    """Blob interpolation function.
    Creates new blobs by interpolating the points on the perimeters of the two inputs.

    Args:
        blob_1: First blob to interpolate
        blob_2: Second blob to interpolate
        zoom: Resolution multiplier for subpixel
        n_points_perimeter: Number of points to sample from each perimeter
        n_interp: Number of blobs to generate
    Returns:
        A list of n_interp blobs
    """
    # Super resolution of the two blobs
    super_1 = get_super_resolution(blob_1, zoom)
    super_2 = get_super_resolution(blob_2, zoom)

    # Perimeter computation
    per_1 = get_perimeter(super_1)
    per_2 = get_perimeter(super_2)

    # Computation of the radial perimeter
    rad_1 = radialize(per_1)
    rad_2 = radialize(per_2)

    # Selects n equidistant points on each radialized perimeter
    points_1 = choose_points(rad_1, n_points_perimeter)
    points_2 = choose_points(rad_2, n_points_perimeter)

    # If registration is required, performs ICP
    if register:
        # Defines 3D point clouds on the z=0 plane
        pcd_points_1 = np.array([[y, x, 0] for y, x in points_1])
        pcd_points_2 = np.array([[y, x, 0] for y, x in points_2])

        # Sets points in Open3D's data structures
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pcd_points_1)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pcd_points_2)

        # Computes the registration and transforms the first blob's points
        reg_p2p = o3d.pipelines.registration.registration_icp(pcd1, pcd2, 10, 
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
        transformed = copy.deepcopy(pcd1)
        transformed.transform(reg_p2p.transformation)
        points_1 = np.asarray(transformed.points)[:, :2]

        # If needed, realigns the blob to the frame (as the registration may translate it)
        min_y = np.min(points_1[:, 0])
        min_x = np.min(points_1[:, 1])
        if min_y < 0:
            points_1[:, 0] += -min_y
        if min_x < 0:
            points_1[:, 1] += -min_x

    # Linear interpolation between points
    lin_spaces = np.array([np.linspace(points_1[i], points_2[i], n_interp) 
                           for i in range(min(len(points_1), len(points_2)))])
    lin_spaces = np.transpose(lin_spaces, (1, 0, 2))

    new_final = []
    for ls in lin_spaces:
        # Define a big-enough boundary s.t. the area closing can operate only on the inside of the blob
        boundary = np.zeros((500, 500))
        # New boundary with removed zoom
        for y, x in ls:
            boundary[round(y / zoom), round(x / zoom)] = 1
        
        # Adding padding for correct area closing
        padded_boundary = np.pad(boundary, [[10, 10], [10, 10]])
        # Blob filling
        filled_mask = morphology.area_closing(padded_boundary, 3000)
        # Dilation and erosion to smooth the mask and remove stray pixels
        new_mask = morphology.dilation(morphology.erosion(filled_mask))
        try:
            # Array reduction to the blob dimension
            awhere = np.argwhere(new_mask)
            y_0 = np.minimum.reduce(awhere[:, 0])
            y_1 = np.maximum.reduce(awhere[:, 0]) + 1
            x_0 = np.minimum.reduce(awhere[:, 1])
            x_1 = np.maximum.reduce(awhere[:, 1]) + 1
            # The final mask is added only if it has enough points and area
            if np.count_nonzero(new_mask) > 50 and np.count_nonzero(new_mask) / ((x_1 - x_0) * (y_1 - y_0)) >= 0.5:
                new_final.append(new_mask[y_0:y_1, x_0:x_1])
        except:
            pass

    if return_lin_spaces:
        return new_final, lin_spaces
    
    return new_final

def plot_interpolated_blobs(interpolated_blobs, out_path, ncols=6):
    """Plot function for interpolated blobs.

    Args:
        interpolated_blobs: the list of interpolated blobs
        out_path: saving path
        n_cols: number of columns in the plot    
    """
    nrows = max(math.ceil(len(interpolated_blobs) / ncols), 2)
    fig, ax = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    i = 0
    for m in interpolated_blobs:
        awhere = np.argwhere(m)
        bbox_width = np.maximum.reduce(awhere[:, 1]) - np.minimum.reduce(awhere[:, 1])
        bbox_height = np.maximum.reduce(awhere[:, 0]) - np.minimum.reduce(awhere[:, 0])
        if np.count_nonzero(m) / (bbox_width * bbox_height) >= 0.5:
            ax[i // ncols, i % ncols].imshow(m)     
        else:
            ax[i // ncols, i % ncols].imshow(m, cmap='hot') 
        i += 1   
    plt.savefig(out_path)
    plt.close()

def save_interp_animation(lin_spaces, out_path):
    """Animation function for interpolation

    Args:
        lin_spaces: The interpolated linear spaces
        out_path: Path for saving
    """
    fig, ax = plt.subplots()
    camera = Camera(fig)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    n_ls = len(lin_spaces)
    for i, ls in enumerate(lin_spaces):
        plt.scatter(ls[:, 1], ls[:, 0], color=cmap_color(i, n_ls, cm.winter, norm=True))
        camera.snap()

    for i, ls in enumerate(lin_spaces[::-1]):
        plt.scatter(ls[:, 1], ls[:, 0], color=cmap_color(i, n_ls, cm.winter_r, norm=True))
        camera.snap()
    
    animation = camera.animate(interval=100)
    animation.save(out_path, dpi=200)

#######################
# Placement functions #
#######################

# def get_prior_map(n_noises, base_octave, shape):
#     octaves = np.array([base_octave * 2 ** i for i in range(0, n_noises)])
#     amps = np.array([1 / (5 ** i) for i in range(0, n_noises)])

#     noises = [PerlinNoise(octaves=o) for o in octaves]
#     noise_mat = np.array([[[noise([i/shape[0], j/shape[1]]) for noise in noises]for j in range(shape[1])] for i in range(shape[0])])
#     noise_map = noise_mat @ amps

#     return noise_map

# def get_prior_map(shape, base, scale, octaves, persistence, lacunarity, repeatx=1024, repeaty=1024):
#     width = shape[1]
#     height = shape[0]

#     prior = np.zeros((height, width))
#     for i in range(height):
#         for j in range(width):
#             prior[i][j] = noise.snoise2(i/scale, j/scale, octaves=octaves, persistence=persistence, 
#                                         lacunarity=lacunarity, repeatx=repeatx, repeaty=repeaty, base=base)
#     return prior

# def tensor_linspace(start, end, steps=10):
#     base_linspace = torch.linspace(0,1,steps)
#     return (base_linspace*(end-start)) + start

# def get_prior_map(size, n_octaves, offsets, amplitudes, frequencies, seed=None):
#     assert len(frequencies) == len(offsets) == len(amplitudes), "Different number of frequencies, offsets, or amplitudes"
#     offsets = torch.abs(offsets)  # to avoid errors with negative offset
#     max_size = max(size)
#     p = torch.zeros((max_size,max_size))
#     for i in range(len(frequencies)):
#         freq = frequencies[i]
#         amplitude = amplitudes[i]
#         offset_x = offsets[i][0]
#         offset_y = offsets[i][1]
#         lin_x = tensor_linspace(offset_x, offset_x+freq, max_size+1)[:-1]  # torch.linspace(offset_x, offset_x+freq, max_size+1)[:-1]
#         lin_y = tensor_linspace(offset_y, offset_y+freq, max_size+1)[:-1]  # torch.linspace(offset_y, offset_y+freq, max_size+1)[:-1]
#         x, y = torch.meshgrid(lin_x, lin_y, indexing="xy") 
#         p += amplitude * perlin(x, y, seed=seed)
#     return p[0:size[1],0:size[0]]

def can_host(base_mask: np.array, cutter_mask: np.array, n: int, m: int) -> bool:
	"""
	Checks if cutter_mask can be placed over base_mask in coordinates (n, m)
	Args:
		base_mask: The binary mask of the available dough
		cutter_mask: The binary mask of the cutter to place
		n: Row where to place cutter_mask
		m: Column where to place cutter_mask
	Returns:
		True if the cutter mask falls entirely in the base_mask
	"""
	in_range = n + cutter_mask.shape[0] <= base_mask.shape[0] and m + cutter_mask.shape[1] <= base_mask.shape[1]
	if not in_range:
		# (n, m) is in OUT_i for cutter_mask i
		return False
	# Overlaps the two masks; if base_mask can entirely host cutter_mask, the result equals cutter_mask
	masks_and = np.logical_and(base_mask[n:n+cutter_mask.shape[0], m:m+cutter_mask.shape[1]] > 0, cutter_mask)
	return np.array_equal(masks_and, cutter_mask)


def create_circle_array(width, height, center_x, center_y, radius):
    y, x = np.ogrid[:height, :width]
    circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    binary_array = np.zeros((height, width), dtype=int)
    binary_array[circle_mask] = 1
    return binary_array


def greedy_placement_spaced(prior_map, blobs, centroids_pdf, centroids_bin_edges, top_percent=0.16, availability_map=None, cost_thresh=0, 
                     exact_order=False, max_time=60):
    """Performs the greedy placement of blobs given their prior_map

    Args:

    Returns:

    """
    new_masks = []
    working_blobs = copy.deepcopy(blobs)
    random.shuffle(working_blobs)
    # Initialization of the availability map (initally all positions can be used)
    if availability_map is None:
        availability_map = np.ones_like(prior_map).astype(bool)
    
    # sum_filter = np.ones((20, 20))
    # # Convolve the cost map to have an estimate of the total cost around each single pixel
    # conv_map = signal.convolve2d(prior_map, sum_filter, mode='same')

    # Restriction to places with enough probability
    availability_map = availability_map & (prior_map > cost_thresh)

    # Parameter for the argpartition
    n_part = int(top_percent * prior_map.shape[0] * prior_map.shape[0])
    found = True
    # The placement stops if max_time is reached
    start = time()
    while found and time() - start < max_time:
        # Choose the position among the available ones
        guiding_map = availability_map * prior_map
        # Top-scoring pixels in the guiding map
           
        # top_choices_i = np.unravel_index(np.argsort(guiding_map.flatten())[::-1], guiding_map.shape)      

        # Computation of the top positions. If exact, orders all positions by their value. Otherwise, chooses one among
        # the top ones
        flat_guiding = guiding_map.flatten()
        top_choices_i_unordered = np.argpartition(flat_guiding, -n_part, axis=None)[-n_part:]
        if exact_order:
            guiding_values = flat_guiding[top_choices_i_unordered]
            sorted_indices = np.argsort(guiding_values.flatten())[::-1]
            top_choices_i = np.unravel_index(top_choices_i_unordered[sorted_indices], guiding_map.shape)
        else:
            top_choices_i = np.unravel_index(top_choices_i_unordered[::-1], guiding_map.shape)
        
        # Search for a blob compatible with one of the top-scoring positions            
        found = False
        blob = None
        start_i = None
        start_j = None
        iter_count = 0
        for y, x in zip(*top_choices_i):            
            mask_i = 0
            # For each (y, x) position looks for a fitting blob
            for b in working_blobs:
                if time() - start > max_time:
                    break
                # Check if the mask can be put centered on the pixel             
                found = y - b.shape[0] // 2 > 0 and y + b.shape[0] // 2 + 1 < guiding_map.shape[0] \
                        and x - b.shape[1] // 2 > 0 and x + b.shape[1] // 2 + 1 < guiding_map.shape[1] \
                        and can_host(guiding_map > 0, b, y - b.shape[0] // 2, x - b.shape[1] // 2)
                if found:
                    # When a fitting blob is found, it is removed from the working blobs and placed
                    blob = b
                    start_i = y - b.shape[0] // 2 
                    start_j = x - b.shape[1] // 2
                    del working_blobs[mask_i]
                    break
                mask_i += 1
            
            if found or len(working_blobs) == 0:
                break
            iter_count += 1
        
        if blob is not None:
            new_mask = np.zeros_like(guiding_map)
            new_mask[start_i:start_i+blob.shape[0], start_j:start_j+blob.shape[1]] = blob
            
            margin = np.round(sample_from_bins(centroids_pdf, centroids_bin_edges)).astype(int)
            expanded_mask = np.copy(new_mask)
            expanded_mask = np.logical_or(expanded_mask, create_circle_array(expanded_mask.shape[1], 
                                                                             expanded_mask.shape[0], y, x, margin))

            availability_map = np.logical_and(availability_map, np.logical_not(expanded_mask))

            new_masks.append(new_mask)

    return new_masks    
    
def plot_placement(masks, prior_map, out_path="placement.png"):
    plt.figure(figsize=(12, 12))
    plt.subplot(121)
    plt.title("Prior map")
    plt.imshow(prior_map)
    plt.subplot(122)
    plt.title("Placed masks")
    plt.imshow(np.where(np.logical_or.reduce(masks), 0, prior_map))
    plt.savefig(out_path)
    plt.close()

################
# Final output #
################

def cmap_color(i: int, n: int, cmap = cm.rainbow, alpha: int = 255, desaturate: bool = False, norm: bool = False):
    """Gets the color associated to element i of n by applying cmap

    Args:
        i (int): The current element's number
        n (int): Total number of elements
        cmap (matplotlib.cmap): Color map to apply
        alpha (int): Alpha channel of the applied color, in [0, 255]
        desaturate (bool): If True, the color gets desaturated

    Returns:
        The color coded in RGBA, in [0, 255]
    """
    color = cmap(i / n)[:3]
    if desaturate:
        h, s, v = rgb_to_hsv(*color)
        color = hsv_to_rgb(h, s / 2, v)

    result = tuple([*(int(c * 255) for c in color), alpha])

    if norm:
        result = [c / 255 for c in result]

    return result

def get_mask_border(mask):
    """Returns true where the mask has its boundary (the border is part of the mask)"""
    return np.logical_and(np.logical_or(*np.gradient(mask * 1)), mask)

def plot_result(prior_map, image_path, masks, output_path):
    pil_image = Image.open(image_path)
    image = np.array(pil_image)

    mask_image = np.zeros((*masks[0].shape[:2], 4))
    border = np.zeros(masks[0].shape[:2])

    n_masks = len(masks)
    for i, mask in enumerate(masks):
        mask = mask[..., None]
        color = cmap_color(i, n_masks, alpha=33)
        mask_image = np.where(mask, color, mask_image)
        border = np.where(get_mask_border(np.squeeze(mask)), 1, border)
    
    mask_image[:, :, 3] = np.where(border, 255, mask_image[:, :, 3])
    mask_image = mask_image.astype(np.uint8)
    target_shape = mask_image.shape[:2]
    base_overlay = pil_image.convert('RGBA').resize(target_shape[::-1])
    mask_image = Image.fromarray(mask_image)
    base_overlay.paste(mask_image, (0, 0), mask_image)
    
    plt.figure(figsize=(10, 10))
    
    plt.subplot(131)
    plt.imshow(image)
    plt.title("Generated image")

    plt.subplot(132)
    plt.imshow(base_overlay)
    plt.title("Mask overlay")

    plt.subplot(133)
    plt.imshow(prior_map)
    plt.title("Prior")

    plt.savefig(output_path, dpi=200)

    plt.close()


# -------------------
# Distance components
# -------------------

def get_blob_centroids(binary_array: np.ndarray):
    """
    Given an array (N_BLOBS, H, W), returns the coordinates of the blobs centroids
    """    
    centroids = []

    for i in range(binary_array.shape[0]):
        centroid_y, centroid_x = center_of_mass(binary_array[i])
        centroids.append([centroid_x, centroid_y])
    
    return np.array(centroids)

def closest_centroid_distance(centroids: np.ndarray, current_channel: int):
    current_centroid = centroids[current_channel:current_channel+1]
    other_centroids = centroids[[i for i in range(centroids.shape[0]) if i != current_channel]]

    # Compute minimum distance between current and other blob coordinates
    dist = np.linalg.norm(current_centroid[np.newaxis, :, :] - other_centroids[:, np.newaxis, :], axis=2)
    min_dist = np.min(dist)

    return min_dist

def get_distance_distribution(centroids: List[np.ndarray], n_bins: int):
    """
    Given a list of B masks (N_BLOBS, H, W), returns a binned distribution of the minimum
    distances between blobs and their closest neighbours.

    Returns a tuple (distr, bin_edges), where distr is the probability density for the respective bin,
    whose edges are defined in array bin_edges with length (n_bins+1).
    """
    # Compute all min distances for all blobs for all masks
    dists = [closest_centroid_distance(centroids[a], i) for a in range(len(centroids)) for i in range(centroids[a].shape[0])]
    distr, bin_edges = np.histogram(dists, bins=n_bins)
    distr = distr.astype(float)
    distr /= np.sum(distr)
    return distr, bin_edges

def sample_from_bins(pdf: np.ndarray, bin_edges: np.ndarray):
    """
    Given an array defining a discrete pdf and an array defining the bin edges to which
    the pdf is associated, returns a number sampled uniformly from a bin sampled according
    to the binned pdf.

    If shape is None, returns a single value, otherwise returns an array with the given shape.
    """
    bin_idx = np.random.choice(len(pdf), p=pdf)
    return np.random.uniform(bin_edges[bin_idx], bin_edges[bin_idx+1], size=None)

def compute_centroids(cache_path, blob_path):
    if not os.path.exists(cache_path):
        print("Computing distances...", end=' ')
        blobs_normal = load_full_blobs_mat_by_image(blob_path)
        centroids = [get_blob_centroids(m) for m in blobs_normal]
        pdf, bin_edges = get_distance_distribution(centroids, n_bins=10)
        pdf_dict = {'pdf': list(pdf), 'bin_edges': list(bin_edges)}
        with open(cache_path, "w") as f:
            json.dump(pdf_dict, f)
    else:
        print("Loading distances...", end=' ')
        with open(cache_path, "r") as f:
            pdf_dict = json.load(f)
        pdf = np.array(pdf_dict['pdf'])
        bin_edges = np.array(pdf_dict['bin_edges'])
    print("Done")
    return pdf, bin_edges

########################################
# Perlin noise optimization components #
########################################

from PIL import ImageFilter
import torch
import optimizator

def get_prior_parameters(params_path, mask_path, n_octaves):
    if not os.path.exists(params_path):
        mask = np.array(sio.loadmat(mask_path)['inst_map'])
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)

        min_size = min(*(np.shape(mask)))

        blurred_mask = Image.fromarray(mask[:min_size, :min_size]).filter(ImageFilter.GaussianBlur(radius=30))
        mask_torch = torch.from_numpy(np.array(blurred_mask))

        config = optimizator.PerlinConfig(mask_torch.shape, n_octaves=n_octaves)
        optimizer = optimizator.PerlinOptimizer(config)

        # INITIALIZATION IS CRUCIAL, DO NOT PUT SYMMETRIC VALUES OR IT WON'T WORK
        trial = {
            'n_octaves': n_octaves,
            'frequencies': torch.tensor([1,2,1,2]).float(),
            'offsets': torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1]]).float(),
            'amplitudes': torch.tensor([0.2, 0.3, 0.2 , 0.3]).float()
        }

        # Optimize parameters
        optimizer_out = optimizer.run_optimization(mask_torch, trial=trial, lr=0.0005, num_iterations=10000)

        with open(params_path, "w") as f:
            json.dump(optimizer_out, f)
    else:
        print("Loading prior parameters...", end=' ')
        with open(params_path, "r") as f:
            optimizer_out = json.load(f)
        print("Done")

    return optimizer_out


