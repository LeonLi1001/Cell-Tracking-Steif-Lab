'''
This script has been tested on BCGSC dlhost12 with CUDA Version: 12.0 
'''

import numpy as np
import pandas as pd
import os 
import sys
from loader import numerical_sort, load_images_from_directory, load_masks_from_directory, remove_empty_frame, load_tif_masks_from_directory
from visualizer import process_frames, create_video
from postprocess import find_small_non_moving_tracks, postprocess_frame, update_track_info_across_frame, predict_next_centroids, maj_object_within_radius, remove_track, diverge_track, merge_track
from features import calculate_features
import warnings
import joblib
import shutil
import tifffile as tiff
from skimage.measure import regionprops
#############################################################
# Section 1: Input Parameters and Configuration
#############################################################
chip = "A138856A" 
run = "10dropRun1" 

import torch
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks
from trackastra.data import example_data_bacteria, example_data_hela, example_data_fluo_3d

device = "cuda" if torch.cuda.is_available() else "cpu"

#############################################################
# Section 2: Initial Tracking with Trackastra
#############################################################
main_img_directory = f"/projects/steiflab/archive/data/imaging/{chip}/NozzleImages/{run}"
#main_img_directory = f"/projects/steiflab/archive/data/imaging/{chip}/CellenONEImages/{run}" # for cellenONE
main_mask_directory = f"/projects/steiflab/scratch/leli/{chip}/{run}/rcnn_output_masks"
out_folder = f'{chip}/{run}/tracked'

# Load images
imgs, img_names = load_images_from_directory(main_img_directory)

# Load masks
masks = load_masks_from_directory(main_mask_directory, imgs[0].shape, fix_overlap = True, overlap_threshold = 0.5)

print("Images shape:", imgs.shape)
print("Masks shape:", masks.shape)

# Ensure the shape matches the required format: (time, y, x)
imgs = imgs.reshape(-1, imgs.shape[1], imgs.shape[2])
masks = masks.reshape(-1, masks.shape[1], masks.shape[2])

imgs, masks, ind_to_remove = remove_empty_frame(imgs, masks)

print("After filtering Images shape:", imgs.shape)
print("After filtering Masks shape:", masks.shape)


# Load a pretrained model
model = Trackastra.from_pretrained("general_2d", device=device)

# Track the cells
track_graph = model.track(imgs, masks, mode="greedy")  # or mode="ilp", or "greedy_nodiv"

# Write to cell tracking challenge format
ctc_tracks, masks_tracked = graph_to_ctc(
      track_graph,
      masks,
      outdir=out_folder,
)

## create a file that connects tiffs with the images
tifs = sorted([t for t in os.listdir(out_folder) if t.endswith(".tif")])
img_names_new = np.delete(img_names, ind_to_remove, axis = 0)
link_file = pd.DataFrame({"tifs": tifs, "imgs": img_names_new})
link_file.to_csv(os.path.join(out_folder, "tif_to_img.csv"), index=False)

# Directory containing tracking results (TIFF files and text file)
tracking_dir = f"/projects/steiflab/scratch/leli/trackastra/{out_folder}"

# Create an output directory for PNGs
output_dir = f"/projects/steiflab/scratch/leli/trackastra/{out_folder}_imgs/"
os.makedirs(output_dir, exist_ok=True)

total_frame_num = len([file for file in os.listdir(main_img_directory) if file.startswith("Image_") and file.endswith(".png")])
# total_frame_num = len([filename for filename in os.listdir(main_img_directory) if filename.endswith("_htert_Run.png") and not filename.startswith("._") and "Printed" not in filename])
print(f"total frame number is {total_frame_num}")

# Process each frame
act_rcnn_inds = [i+1 for i in range(total_frame_num) if i not in ind_to_remove] # here we are tying to find the corresponding frame index that matches with the rcnn results
assert len([file for file in os.listdir(tracking_dir) if not file.startswith("._") and file.endswith("tif")]) == len(act_rcnn_inds)

process_frames(imgs, tracking_dir, output_dir, frames_to_process = None)
print(f"Process frames done!")

output_video = f'/projects/steiflab/scratch/leli/trackastra/{out_folder}_imgs/tracked_video_full.mp4'
height, width = imgs.shape[1], imgs.shape[2]
create_video(output_dir, output_video, imgs.shape[0], width, height, fps=3, frames_to_process = None)

print("#### Section 2: Input Parameters and Configuration ####: COMPLETE")

#############################################################
# Section 3: Immobile Object Filtering
#############################################################
out_folder = f'{chip}/{run}/tracked'
track_info_file = f'/projects/steiflab/scratch/leli/trackastra/{out_folder}/man_track.txt'
tif_directory = f"/projects/steiflab/scratch/leli/trackastra/{out_folder}"
linkfile_directory = f"/projects/steiflab/scratch/leli/trackastra/{out_folder}/tif_to_img.csv"
out_folder = f'{chip}/{run}/tracked_again'

# get the track ids that should be removed
small_non_moving_tracks = find_small_non_moving_tracks(track_info_file, tif_directory)
print(f"the small no moving objects are : {small_non_moving_tracks}")

# Load images
linkfile = pd.read_csv(linkfile_directory)
imgs, img_names = load_images_from_directory(main_img_directory, list(linkfile['imgs']))

# Load masks
masks = load_tif_masks_from_directory(tif_directory, imgs[0].shape, small_non_moving_tracks)

print("Images shape:", imgs.shape)
print("Masks shape:", masks.shape)

# Ensure the shape matches the required format: (time, y, x)
imgs = imgs.reshape(-1, imgs.shape[1], imgs.shape[2])
masks = masks.reshape(-1, masks.shape[1], masks.shape[2])

imgs, masks, ind_to_remove = remove_empty_frame(imgs, masks)

print("Images shape:", imgs.shape)
print("Masks shape:", masks.shape)


# Load a pretrained model
# or from a local folder
# model = Trackastra.from_folder('path/my_model_folder/', device=device)
model = Trackastra.from_pretrained("general_2d", device=device)

# Track the cells
track_graph = model.track(imgs, masks, mode="greedy")  # or mode="ilp", or "greedy_nodiv"

# Write to cell tracking challenge format
ctc_tracks, masks_tracked = graph_to_ctc(
    track_graph,
    masks,
    outdir=out_folder,
)

## create a file that connects tiffs with the images
tifs = sorted([t for t in os.listdir(out_folder) if t.endswith(".tif") and not t.startswith("._")])
img_names_new = np.delete(img_names, ind_to_remove, axis = 0)
link_file = pd.DataFrame({"tifs": tifs, "imgs": img_names_new})
link_file.to_csv(os.path.join(out_folder, "tif_to_img.csv"), index=False)

# Directory containing tracking results (TIFF files and text file)
tracking_dir = f"/projects/steiflab/scratch/leli/trackastra/{out_folder}"
print(f"{os.path.isdir(tracking_dir)}")
# Create an output directory for PNGs
output_dir = f"/projects/steiflab/scratch/leli/trackastra/{out_folder}_imgs/"
os.makedirs(output_dir, exist_ok=True)

total_frame_num = len([file for file in os.listdir(main_img_directory) if file.startswith("Image_") and file.endswith(".png")])
#total_frame_num = len([filename for filename in os.listdir(main_img_directory) if filename.endswith("_htert_Run.png") and not filename.startswith("._") and "Printed" not in filename])
print(f"total frame number is {total_frame_num}")

# Process each frame
act_rcnn_inds = [i+1 for i in range(total_frame_num) if i not in ind_to_remove] # here we are tying to find the corresponding frame index that matches with the rcnn results
#assert len([file for file in os.listdir(tracking_dir) if not file.startswith("._") and file.endswith("tif")]) == len(act_rcnn_inds)

process_frames(imgs, tracking_dir, output_dir, frames_to_process = None)
print(f"Process frames done!")

output_video = f'/projects/steiflab/scratch/leli/trackastra/{out_folder}_imgs/tracked_video_full.mp4'
height, width = imgs.shape[1], imgs.shape[2]
create_video(output_dir, output_video, imgs.shape[0], width, height, fps=3, frames_to_process = None)

print("#### Section 3: Immobile Object Filtering ####: COMPLETE")

#############################################################
# Section 4: Sub Track Processing
#############################################################
# Define paths
source_tif_dir = f"/projects/steiflab/scratch/leli/trackastra/{out_folder}"
target_tif_dir = f"/projects/steiflab/scratch/leli/trackastra/{out_folder}_postprocessed"
track_info_file = f'/projects/steiflab/scratch/leli/trackastra/{out_folder}/man_track.txt'
classifier_model_path = f'/projects/steiflab/scratch/leli/trackastra/postprocessing/models/Random Forest_model.pkl'
example_csv_path = f"/projects/steiflab/scratch/leli/trackastra/postprocessing/df.csv"

os.makedirs(target_tif_dir, exist_ok=True)
# Copy original tif files to the target directory
if os.path.exists(target_tif_dir):
    shutil.rmtree(target_tif_dir)
shutil.copytree(source_tif_dir, target_tif_dir)

os.makedirs(target_tif_dir, exist_ok=True)


# Load the classifier model
classifier = joblib.load(classifier_model_path)
#classifier = test_model['SVM']

# Read the track info and example CSV
track_info = pd.read_csv(track_info_file, sep='\s+', names=['Track_ID', 'Start', 'End', 'Parent'])
new_track_info = pd.DataFrame(columns=track_info.columns)

# here we are finding the root track of all
def find_root(track_id):
    parent = track_info.loc[track_info['Track_ID'] == track_id, 'Parent'].values[0]
    if parent == 0:
        return track_id
    else:
        return find_root(parent)

track_info['Root'] = track_info['Track_ID'].apply(find_root)

# here we are prepping and predicting the action classes for all
track_info_file = f'/projects/steiflab/scratch/leli/trackastra/{out_folder}/man_track.txt'
tif_directory = f'/projects/steiflab/scratch/leli/trackastra/{out_folder}'
track_ids = track_info.loc[track_info['Parent'] != 0]['Track_ID']
features_dict = calculate_features(track_info_file, tif_directory, track_ids)
if not features_dict['Track_ID']: 
    print("there are no sub tracks!!!")
    sys.exit()

examples = pd.DataFrame(features_dict)
examples["action"] = classifier.predict(examples)
examples = examples.set_index('Track_ID')

for filename in os.listdir(target_tif_dir):
    if not filename.startswith("._") and filename.endswith(".tif"):
        frame_path = os.path.join(target_tif_dir, filename)
        frame = tiff.imread(frame_path)
        print(f"start processing frame: {filename} #####################################")
        processed_frame = postprocess_frame(frame, track_info, examples)
        tiff.imwrite(frame_path, processed_frame)
        print(f"complete processing frame: {filename}")

        frame_num = int(filename.replace('man_track', '').replace('.tif', ''))
        new_track_info = update_track_info_across_frame(track_info, new_track_info, processed_frame, frame_num)
        new_track_info.to_csv(os.path.join(target_tif_dir, "man_track.txt"), sep=' ', index=False, header=False)

        #if frame_num == 1032: # and frame_num <= 2519:
            #print(f"1032 has unique values")
            #display_colored_images(processed_frame, labels_to_color = [178, 189, 190, 191], title= f'current Frame: {frame_num}')
            #break
            #print(f"the frame 2517 has unique values: {np.unique(processed_frame)}")
             
            #plot_frame(frame, title=f'current Frame: {frame_num} when track id is: {track_id}', ids = [track_id]+siblings)

print("Sub track Processing completed.")

# Directory containing tracking results (TIFF files and text file)
tracking_dir = f"/projects/steiflab/scratch/leli/trackastra/{out_folder}_postprocessed"
print(f"{os.path.isdir(tracking_dir)}")
# Create an output directory for PNGs
output_dir = f"/projects/steiflab/scratch/leli/trackastra/{out_folder}_postprocessed_imgs"
os.makedirs(output_dir, exist_ok=True)

# Process each frame
total_frame_num = len([file for file in os.listdir(main_img_directory) if file.startswith("Image_") and file.endswith(".png")])
#total_frame_num = len([filename for filename in os.listdir(main_img_directory) if filename.endswith("_htert_Run.png") and not filename.startswith("._") and "Printed" not in filename])

process_frames(imgs, tracking_dir, output_dir, frames_to_process = None)
print(f"Process frames done!")

output_video = f'/projects/steiflab/scratch/leli/trackastra/{out_folder}_postprocessed_imgs/tracked_video_full.mp4'
create_video(output_dir, output_video, imgs.shape[0], width, height, fps=3, frames_to_process = None)

print("#### Section 4: Sub Track Processing ####: COMPLETE")

#############################################################
# Section 5: ALL Track Processing
#############################################################
import warnings
import joblib
import shutil

# Define paths
#out_folder = "A138974A/PrintRun_Apr1223_1311/tracked"
source_tif_dir = f"/projects/steiflab/scratch/leli/trackastra/{out_folder}_postprocessed"
target_tif_dir = f"/projects/steiflab/scratch/leli/trackastra/{out_folder}_postprocessed_2.0"
track_info_file = f'/projects/steiflab/scratch/leli/trackastra/{out_folder}_postprocessed_2.0/man_track.txt'


os.makedirs(target_tif_dir, exist_ok=True)
# Copy original tif files to the target directory
if os.path.exists(target_tif_dir):
    shutil.rmtree(target_tif_dir)
shutil.copytree(source_tif_dir, target_tif_dir)

print("Initiate Main track Processing ... ")
track_info = pd.read_csv(track_info_file, sep='\s+', names=['Track_ID', 'Start', 'End', 'Parent']) # read in again the new versin of track info 
track_ids = sorted(track_info['Track_ID'])
#track_ids  = [50]

# This number is the original max track id, for divered track we would want to creat new ones
new_track_label = np.max(track_info['Track_ID'].values)

while track_ids: 
    
    print(f"Check for track ID: {track_ids[0]}")

    # there may be changes evertime we go throuh a track
    track_info = pd.read_csv(track_info_file, sep='\s+', names=['Track_ID', 'Start', 'End', 'Parent']) # read in again the new versin of track info 

    #### Check 1: <=2 Frames
    track_id, start_frame, end_frame, parent_id = track_info.loc[track_info['Track_ID'] == track_ids[0]].values[0]
    if end_frame - start_frame <=2: 
        remove_track(track_info_file, target_tif_dir, to_be_removed_id = track_ids[0])
        track_ids = track_ids[1:]
        continue

    #### Check 2: Moving up Tracks and paused tracks 
    track_id, start_frame, end_frame, parent_id = track_info.loc[track_info['Track_ID'] == track_ids[0]].values[0]
    centroids = []
    centroids_frame = []
    skipped_frames = []
    for frame_number in range(start_frame, end_frame+1):

        #print(f"centroids_with_prediction is : {centroids_with_prediction}")

        frame_path = os.path.join(target_tif_dir, f'man_track{frame_number:04d}.tif')
        frame = tiff.imread(frame_path)
        binary_mask = (frame == track_id).astype(np.uint8)

        #When this track does not exist in this frame we keep going 
        if len(np.unique(binary_mask)) == 1:
            skipped_frames.append(frame_number)
            continue

        # add in the centroid
        centroid = regionprops(binary_mask)[0].centroid
        if centroid is None: raise ValueError("The centroid point being added is Nnne")

        if len(centroids) >=5: # if we are in the middle of the tracklet 
            y_changes = np.diff([c[0] for c in centroids])
            median_change = np.median(y_changes)
            mad = np.median(np.abs(y_changes - median_change))
            threshold = 2.5 * mad # 2.5 is the usual value but can be changed

            #print(f"The centroids are currently {centroids}")

            # here we already have the MAD threshold, the lower bound is median change in y direction - the threshold. 
            # Since the object is always going down, the y value should only increase. So once the object move up, the change in y value should be negative so it is on the lower bound. 
            # Here we are checking if it is outside the lower bound. 
            #print(f"The difference between the current object and the last one is {np.diff([centroids[-1][0], centroid[0]])} with the last item being {centroids[-1][0]} and the current centroid y value is {centroid[0]} with the lower bound be {median_change - threshold}")
            if np.diff([centroids[-1][0], centroid[0]]) <= median_change - threshold: 

                # since this is case 2 so we add a prefix to the track id so we can come back to it
                new_track_label = new_track_label+1
                diverge_track(track_info_file, target_tif_dir, to_be_split_id = track_id, new_id = int(new_track_label), diverging_start_frame = frame_number)
                track_ids.append(int(new_track_label))
                break

            elif len(skipped_frames) >=2:

                # since this is case 2.5 so we add a prefix to the track id so we can come back to it
                new_track_label = new_track_label+1
                diverge_track(track_info_file, target_tif_dir, to_be_split_id = track_id, new_id = int(new_track_label), diverging_start_frame = frame_number)
                track_ids.append(int(new_track_label))
                break
        
        centroids.append(centroid)
        centroids_frame.append(frame_number)


        # else: # if we are at the beginning we do not do anything yet, might change later

    #### Check 3: label switching
    centroids_with_prediction = centroids.copy() # here this centroid will contain the LR predicted centroid
    centroids_frame_with_prediction = centroids_frame.copy()

    #print(f"IN CASE 3: The centroids are {centroids_with_prediction} and the frame numbers are {centroids_frame_with_prediction}")

    if len(centroids_with_prediction) >= 5: 
        covered_by = []
        
        for frame_number in range(sorted(centroids_frame_with_prediction, reverse = True)[0] + 1, sorted(centroids_frame_with_prediction, reverse = True)[0] + 4):
            frame_path = os.path.join(target_tif_dir, f'man_track{frame_number:04d}.tif')
            if os.path.exists(frame_path):
                frame = np.array(tiff.imread(frame_path))

                curr_c = predict_next_centroids(centroids_with_prediction, centroids_frame_with_prediction, predict_this_frame = frame_number)

                #centroids_with_prediction.append(curr_c)
                #centroids_frame_with_prediction.append(frame_number)

                if 0 <= int(curr_c[0]) < frame.shape[0] and 0 <= int(curr_c[1]) < frame.shape[1]:
                    maj_label = maj_object_within_radius(frame, curr_c, radius = 3.5)
                    covered_by.append(maj_label)
                    if maj_label != 0:
                        binary_mask = (frame == maj_label).astype(np.uint8)
                        centroid = regionprops(binary_mask)[0].centroid
                        centroids_with_prediction.append(curr_c)
                        centroids_frame_with_prediction.append(frame_number)

                else:
                    print("prediction went out of bound")
                    covered_by.append(0)
                    centroid = regionprops(binary_mask)[0].centroid


        #print(f"here we see that the track is covered by {covered_by} when the centroids are {centroids_with_prediction}")
        if len(covered_by) == 3:
            non_zero_values = [x for x in covered_by if x > 0]
            for label in set(non_zero_values):
                if non_zero_values.count(label) >= 2:
                    # the ids that are to be merge do not matter because we pick which one is which within the merge track function
                    new = merge_track(track_info_file, target_tif_dir, to_be_merged_ids = (label, track_id))

                    temp = new["Track_ID"].values
                    print(f"The label {label} is not in the track info {label not in temp} and the track id {track_id}")
                    if label not in new["Track_ID"].values and label in track_ids: # make sure to have .values, apparently pd series check index not the value if we do no include this :(
                        track_ids.remove(label)

    #### Check 4: Overall y movement to remove the ones that did not make a movement 
    if len(centroids) < 5 and len(centroids) > 1:
        overall_y_movement = centroids[-1][0] - centroids[0][0]
        if overall_y_movement <= 5:
            remove_track(track_info_file, target_tif_dir, to_be_removed_id = track_ids[0])


    assert len(track_ids) == len(np.unique(track_ids))

    track_ids = track_ids[1:]


print("Main track Processing completed.")

print("initiate updating tracking info csv at the very end to correct and ensure the track info final version")

track_info = pd.read_csv(track_info_file, sep='\s+', names=['Track_ID', 'Start', 'End', 'Parent']) # read in again the new versin of track info 
new_track_info = pd.DataFrame(columns=track_info.columns)
for filename in os.listdir(target_tif_dir):
    if not filename.startswith("._") and filename.endswith(".tif"):
        frame_path = os.path.join(target_tif_dir, filename)
        frame = tiff.imread(frame_path)

        frame_num = int(filename.replace('man_track', '').replace('.tif', ''))
        new_track_info = update_track_info_across_frame(track_info, new_track_info, frame, frame_num)
        new_track_info.to_csv(os.path.join(target_tif_dir, "man_track.txt"), sep=' ', index=False, header=False)

# Directory containing tracking results (TIFF files and text file)
tracking_dir = f"/projects/steiflab/scratch/leli/trackastra/{out_folder}_postprocessed_2.0"
print(f"{os.path.isdir(tracking_dir)}")
# Create an output directory for PNGs
output_dir = f"/projects/steiflab/scratch/leli/trackastra/{out_folder}_postprocessed_2.0_imgs"
os.makedirs(output_dir, exist_ok=True)

# Process each frame
total_frame_num = len([file for file in os.listdir(main_img_directory) if file.startswith("Image_") and file.endswith(".png")])
#total_frame_num = len([filename for filename in os.listdir(main_img_directory) if filename.endswith("_htert_Run.png") and not filename.startswith("._") and "Printed" not in filename])

process_frames(imgs, tracking_dir, output_dir, frames_to_process = None)
print(f"Process frames done!")

height, width = imgs.shape[1], imgs.shape[2]  # Get height and width from images
output_video = f'/projects/steiflab/scratch/leli/trackastra/{out_folder}_postprocessed_2.0_imgs/tracked_video_full.mp4'
create_video(output_dir, output_video, imgs.shape[0], width, height, fps=3, frames_to_process = None)

print("#### Section 5: All Track Processing ####: COMPLETE")

