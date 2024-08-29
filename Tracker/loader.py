import os
import cv2
import numpy as np
from skimage import measure
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt, center_of_mass
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from PIL import Image
import re
import pandas as pd
import tifffile as tiff

def numerical_sort(value):
    """
    Extracts the numeric part from the filename for sorting.
    Assumes that the filename format is '<number>_htert_Run'.
    """
    parts = re.findall(r'\d+', value)
    return int(parts[0]) if parts else value

def load_images_from_directory(directory, filtered_img_list = None):
    images = []
    #filenames = sorted([filename for filename in os.listdir(directory) if filename.endswith("_htert_Run.png") and not filename.startswith("._") and "Printed" not in filename], key = numerical_sort) # for cellenONE
    if filtered_img_list: 
        filenames = sorted([filename + ".png" for filename in filtered_img_list], key = numerical_sort)
    else: 
        filenames = sorted([filename for filename in os.listdir(directory) if filename.endswith(".png") and not filename.startswith("._")], key = numerical_sort)
    # print(f"filenames are {filenames}")

    for filename in filenames:
        img_path = os.path.join(directory, filename)
        img = Image.open(img_path).convert('L')
        #img = expand_image(img, mode = "images")
        img_array = np.array(img)
        #img_array = np.rot90(img_array)
        images.append(img_array) # for cellenONE
    return np.array(images), [f.replace(".png", "") for f in filenames]


def expand_image(img, mode, factor=3,):
    # Get original dimensions
    original_width, original_height = img.size

    # Calculate new dimensions
    new_width = original_width * factor
    new_height = original_height * factor

    # Resize the image
    if mode == "images": 
        new_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    elif mode == "masks":
        new_img = img.resize((new_width, new_height), Image.Resampling.NEAREST)

    return new_img

def remove_empty_frame(imgs, masks):
    ind_to_remove = []
    for i in range(masks.shape[0]):
        if np.all(masks[i] == 0):
            ind_to_remove.append(i)

    imgs_new = np.delete(imgs, ind_to_remove, axis = 0)
    masks_new = np.delete(masks, ind_to_remove, axis = 0)

    assert imgs_new.shape == masks_new.shape

    return imgs_new, masks_new, ind_to_remove


def mask_to_bbox(mask):
    """
    Converts a binary mask to a bounding box.

    :param numpy.ndarray mask: Binary mask.
    :return: Bounding box in the format (x, y, w, h).
    :rtype: list[int]
    """
    rows, cols = np.where(mask == 255)
    x1, x2 = np.min(cols), np.max(cols)
    y1, y2 = np.min(rows), np.max(rows)
    return [x1, y1, x2-x1, y2-y1]


def load_masks_from_directory(directory, img_shape, fix_overlap=False, overlap_threshold = 0.6, top_threshold = 0.005, bottom_threshold = 0.995):
    """
    Load masks from a directory and handle overlaps if specified.

    Parameters:
    - directory: str, path to the directory containing the masks.
    - img_shape: tuple, shape of the images.
    - fix_overlap: bool, whether to fix overlaps between masks.

    Returns:
    - masks: numpy array, combined masks for each frame.
    """
    masks = []
    current_object_index = 1
    # frames = sorted([frame for frame in os.listdir(directory) if os.path.isdir(os.path.join(directory, frame)) and frame.endswith("_htert_Run")], key = numerical_sort) # for cellenONE
    frames = sorted([frame for frame in os.listdir(directory) if os.path.isdir(os.path.join(directory, frame)) and frame.startswith("Image_")], key = numerical_sort)
    # print(f"The frames are {frames}")

    for frame in frames: #sorted(os.listdir(directory), key = numerical_sort):  # Loop through the frame folders
        frame_dir = os.path.join(directory, frame)
        if os.path.isdir(frame_dir) and frame.startswith("Image_"):  # The directory needs to start with Image for normal runs 
        # if os.path.isdir(frame_dir) and frame.endswith("_htert_Run"): # for cellenONE
            frame_mask = np.zeros(img_shape, dtype=np.int32)

            if fix_overlap:
                curr_masks = []
                
                for filename in sorted(os.listdir(frame_dir)):
                    if filename.endswith(".png") and not filename.startswith("._"):  # Loop through the png files 
                        mask_path = os.path.join(frame_dir, filename)
                        mask = Image.open(mask_path).convert('L')
                        mask_array = np.array(mask)
                        bbox = mask_to_bbox(mask_array)
                        _, y, _, h = bbox

                        if y >= top_threshold * mask_array.shape[0] and (y + h) <= bottom_threshold * mask_array.shape[0]:  # Ignore detections close to top and bottom thresholds
                            if len(np.unique(mask_array)) != 2:
                                raise ValueError("something is up", np.unique(mask_array))
                            curr_masks.append(mask_array)
                        #else:
                            #print(f"Mask {filename} ignored due to top/bottom threshold")


                #print(f"Number of masks in the current frame: {len(curr_masks)}")

                # Create an overlap matrix
                overlap_matrix = np.zeros((len(curr_masks), len(curr_masks)), dtype=int)
                for i in range(len(curr_masks)):
                    for j in range(i + 1, len(curr_masks)):
                        mask_i = curr_masks[i]
                        mask_j = curr_masks[j]

                        # Ensure masks are binary
                        mask1_binary = (mask_i == 255)
                        mask2_binary = (mask_j == 255)

                        # Calculate the size of each mask
                        size1 = np.sum(mask1_binary)
                        size2 = np.sum(mask2_binary)

                        # Identify the smaller and larger masks
                        if size1 < size2:
                            smaller_mask = mask1_binary
                            larger_mask = mask2_binary
                            smaller_size = size1
                        else:
                            smaller_mask = mask2_binary
                            larger_mask = mask1_binary
                            smaller_size = size2

                        # Calculate the overlap
                        overlap = np.sum(smaller_mask & larger_mask)
                        overlap_percentage = overlap / smaller_size
                        #print(f"overalp between mask {i} and maks {j} is {overlap_percentage} with threshold being {overlap_threshold}")

                        if overlap_percentage >= overlap_threshold:
                            overlap_matrix[i, j] = 1
                            overlap_matrix[j, i] = 1

                #print(f"Overlap matrix:\n{overlap_matrix}")

                # Cluster overlapping objects
                sparse_matrix = csr_matrix(overlap_matrix)
                n_components, labels = connected_components(csgraph=sparse_matrix, directed=False, return_labels=True)

                # Group masks by their component labels to form clusters
                clusters = [[] for _ in range(n_components)]
                for mask_index, component_label in enumerate(labels):
                    clusters[component_label].append(mask_index)

                #print(f"Clusters: {clusters}")

                for c in range(len(clusters)):  # Loop through each cluster and merge them
                    masks_in_cluster = [curr_masks[j] for j in clusters[c]]

                    # Create a combined mask
                    combined_mask = np.zeros_like(masks_in_cluster[0], dtype=np.int32)
                    for mask in masks_in_cluster:
                        combined_mask[mask > 0] = 1

                    # Label the combined mask
                    frame_mask[combined_mask > 0] = current_object_index
                    current_object_index += 1

                    #print(f"Processed cluster {c} with {len(masks_in_cluster)} masks.")

            else:  # If not fixing overlaps
                for filename in sorted(os.listdir(frame_dir)):
                    if filename.endswith(".png") and not filename.startswith("._"):  # Loop through the png files  _htert_Run
                        mask_path = os.path.join(frame_dir, filename)
                        mask = Image.open(mask_path).convert('L')
                        mask_array = np.array(mask)
                        if len(np.unique(mask_array)) != 2:
                            raise ValueError("something is up", np.unique(mask_array))
                        # Assign the current object index to the mask pixels
                        frame_mask[(mask_array == 255)] = current_object_index
                        current_object_index += 1
                        #print(f"Processed mask {filename} with index {current_object_index - 1}")

            masks.append(frame_mask)
            # if np.all(frame_mask == 0): print(f"The frame that has all zero is {frame}")
            #print(f"Added frame mask for {frame}, current number of masks: {len(masks)}")

    #print(f"Total frames processed: {len(masks)}")
    return np.array(masks)

def load_tif_masks_from_directory(directory, img_shape, small_non_moving_tracks):
    masks = []
    current_object_index = 1
    for file in sorted(os.listdir(directory)):
        if file.startswith("._") or not file.endswith(".tif"): continue
        tif = tiff.imread(os.path.join(directory, file))
        if tif is None:
            print(f"Error reading label image: {tif}")
            continue
        unique_labels = sorted(np.unique(tif)) # get the unique labels for this specific frame

        for label in unique_labels: 
            if label == 0:
                continue
            elif label in small_non_moving_tracks: 
                tif[tif == label] = 0
                continue
            tif[tif == label] = current_object_index
            current_object_index += 1

        masks.append(tif)
        
    return np.array(masks)