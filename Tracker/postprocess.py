#############################################################
# Postprocess for sub tracks 
#############################################################

import os
import shutil
import pandas as pd
import numpy as np
from skimage import io
from skimage.measure import label, regionprops
from scipy.ndimage import label
import joblib
import tifffile as tiff
from skimage.morphology import binary_erosion
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed

def connect_objects_localized(mask1, mask2, kernel_size=5, iterations=1):
    """
    Connects objects in two masks using a localized morphological dilation.

    Parameters:
    - mask1: numpy array, the first mask.
    - mask2: numpy array, the second mask.
    - kernel_size: int, size of the dilation kernel.
    - iterations: int, number of dilation iterations.

    Returns:
    - connected_labels: numpy array, the labeled image after connecting objects.
    """
    combined_mask = np.maximum(mask1, mask2)
    binary_mask = (combined_mask > 0)
    distance = distance_transform_edt(binary_mask)
    markers, _ = label(binary_mask)
    labels = watershed(-distance, markers, mask=binary_mask)
    connection_mask = np.zeros_like(combined_mask)
    for label_val in np.unique(labels):
        if label_val == 0:
            continue
        component_mask = (labels == label_val)
        if np.sum(component_mask & mask1) > 0 and np.sum(component_mask & mask2) > 0:
            connection_mask[component_mask] = 255
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_connection = binary_erosion(connection_mask, kernel).astype(np.uint8) * 255
    connected_objects = np.where(eroded_connection > 0, eroded_connection, combined_mask)
    return connected_objects

def get_centroid_y(mask, label_value):
    regions = regionprops(mask)
    for region in regions:
        if region.label == label_value:
            return region.centroid[1]  # Corrected to return the y-coordinate
    return None

def update_track_info_across_frame(old_track_info_df, track_info_df, frame, frame_number):
    unique_labels = np.unique(frame)
    unique_labels = unique_labels[unique_labels != 0]  # Remove background (label 0)

    for track_id in unique_labels:
        row = track_info_df[track_info_df['Track_ID'] == track_id]

        if not row.empty:
            start_frame = int(row['Start'].values[0])
            end_frame = int(row['End'].values[0])
            start_frame = min(start_frame, frame_number)
            end_frame = max(end_frame, frame_number)
            track_info_df.loc[track_info_df['Track_ID'] == track_id, 'Start'] = start_frame
            track_info_df.loc[track_info_df['Track_ID'] == track_id, 'End'] = end_frame
        else:
            parent_value = old_track_info_df.loc[old_track_info_df['Track_ID'] == track_id, 'Parent'].values[0] if track_id in old_track_info_df['Track_ID'].values else 0
            new_row = pd.DataFrame({
                'Track_ID': [track_id],
                'Start': [frame_number],
                'End': [frame_number],
                'Parent': [parent_value]
            })
            track_info_df = pd.concat([track_info_df, new_row], ignore_index=True)
    
    return track_info_df

import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2

def color_labels(tif, labels, colors):
    """
    Color the specified labels in the TIFF image with the given colors.

    Parameters:
    - tif: numpy array, the TIFF image.
    - labels: list of int, the labels to color.
    - colors: list of tuple, the colors corresponding to each label.

    Returns:
    - colored_img: numpy array, the image with colored labels.
    """
    colored_img = cv2.cvtColor(tif.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for label, color in zip(labels, colors):
        mask = (tif == label)
        colored_img[mask] = color

    return colored_img

def display_colored_images(frame, labels_to_color, title):
    """
    Display the colored images for the specified labels
    """
    plt.figure(figsize=(8, 8))
    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255), # Magenta
        (255, 165, 0),  # Orange
        (128, 0, 128) ,  # Purple
    ]

    colored_img1 = color_labels(frame, labels_to_color, colors)

    plt.imshow(colored_img1)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def postprocess_frame(frame, track_info, classification, min_size=100):
    
    groups = track_info.groupby('Root')['Track_ID'].apply(list).to_dict()
    unique_labels = np.unique(frame)
    unique_labels = unique_labels[unique_labels != 0]

    groups = {root: tracks for root, tracks in groups.items() if any(track in unique_labels for track in tracks)}

    for root, group in groups.items():
        print(f"This group has root {root} and contains {group}")

        objs_to_be_merged = []

        for track_id in group:
            print(f"iteration: {track_id} where it is \n {track_info.loc[track_info['Track_ID'] == track_id]}")
            if track_id == root or track_info.loc[track_info['Track_ID'] == track_id, 'Parent'].values[0] == 0:
                continue
            print(f"iteration: {track_id}")
            mask = (frame == track_id)
            print(f"This objects {track_id} has a size of {np.sum(mask)}")

            # Remove small objects
            if np.sum(mask) < min_size:
                frame[mask] = 0
                print("Object with track id: {track_id} is too small so we remove all together")
                continue

            action = classification.loc[track_id, 'action']

            if action == 1:
                print(f"Object with track id: {track_id} has action 1")
                continue  # Keep as is

            elif action == 2:
                print(f"Object with track id: {track_id} has action 2")
                frame[mask] = 0  # Remove

            elif action == 0:
                print(f"Object with track id: {track_id} has action 0")
                centroid_y = get_centroid_y(frame, track_id)
                objs_to_be_merged.append((track_id, centroid_y))
        
        if len(objs_to_be_merged) != 0:
            objs_to_be_merged.sort(key=lambda x: x[1])
            print(f"These are the objects that will be merged {objs_to_be_merged}")

            combined_mask = (frame == objs_to_be_merged[0][0])
            for i in range(1, len(objs_to_be_merged)):
                        next_mask = (frame == objs_to_be_merged[i][0])

                        # here this is becuase we realize that the function morphologyEx alter other part of the masks so it does not just add the connecting part it alters the original
                        old_mask1 = combined_mask.copy()
                        old_mask2 = next_mask.copy()

                        for kernel_size in range(4):
                            temp = np.logical_or(combined_mask, next_mask).astype(np.uint8)
                            combined_mask = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, np.ones((kernel_size * 5, kernel_size * 5), np.uint8))
                            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
                            if num_labels-1 == 1: 
                                break
                        
                        combined_mask = np.logical_or(combined_mask, old_mask1).astype(np.uint8)
                        combined_mask = np.logical_or(combined_mask, old_mask2).astype(np.uint8)
                        
                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
                        if num_labels-1 > 1:
                            #display_colored_images(combined_mask, labels_to_color = [1, 2, 3, 4], title= f'current Frame: {frame_num}')
                            print("Failed to connect objects after multiple attempts.")
                        else: 
                            print(f"Succeeded to connect objects with kernel size {kernel_size}")

            # do the final painting of pixels 
            frame[combined_mask > 0] = root

    return frame

import pandas as pd
from sklearn.linear_model import LinearRegression

def predict_next_centroids(centroids, centroids_frame_with_prediction, predict_this_frame):
    """
    Predicts the next centroid positions given a list of centroids and the corresponding frame numbers.

    Parameters:
    - centroids: list of tuples (y, x) representing the coordinates of the centroids.
    - centroids_frame_with_prediction: list of frame numbers corresponding to the centroids.

    Returns:
    - predicted_centroids: tuple (y, x) representing the predicted coordinates of the next centroid.
    """
    # Create a DataFrame for easy handling
    df = pd.DataFrame(centroids, columns=['y', 'x'])
    df['frame'] = centroids_frame_with_prediction

    # Extract features and targets
    X = df['frame'].values.reshape(-1, 1)  # Frames as features
    y_y = df['y'].values  # y-coordinates as target
    y_x = df['x'].values  # x-coordinates as target

    # Fit linear regression models
    model_y = LinearRegression()
    model_x = LinearRegression()
    model_y.fit(X, y_y)
    model_x.fit(X, y_x)

    # Predict the next frame
    next_frame = np.array([predict_this_frame]).reshape(-1, 1)
    pred_y = model_y.predict(next_frame)[0]
    pred_x = model_x.predict(next_frame)[0]

    return pred_y, pred_x

def remove_track(track_info_file, target_tif_dir, to_be_removed_id):

    track_info = pd.read_csv(track_info_file, sep='\s+', names=['Track_ID', 'Start', 'End', 'Parent'])
    track_id, start_frame, end_frame, parent_id = track_info.loc[track_info['Track_ID'] == to_be_removed_id].values[0]

    for frame_number in range(start_frame, end_frame + 1):
        frame_path = os.path.join(target_tif_dir, f'man_track{frame_number:04d}.tif')
        if os.path.exists(frame_path):
            frame = tiff.imread(frame_path)
            frame[frame == to_be_removed_id] = 0
            tiff.imwrite(frame_path, frame)
            print(f"Removed track {to_be_removed_id} from frame {frame_number}")

    new_track_info = track_info[track_info['Track_ID'] != to_be_removed_id]
    new_track_info.to_csv(track_info_file, sep=' ', index=False, header=False)
    
    return new_track_info

def diverge_track(track_info_file, target_tif_dir, to_be_split_id, new_id, diverging_start_frame):

    track_info = pd.read_csv(track_info_file, sep='\s+', names=['Track_ID', 'Start', 'End', 'Parent'])
    track_id, start_frame, end_frame, parent_id = track_info.loc[track_info['Track_ID'] == to_be_split_id].values[0]

    if diverging_start_frame <= start_frame or diverging_start_frame > end_frame:
        raise ValueError("Diverging start frame must be within the original track's range.")
    
    # Iterate through the frames and update the track
    for frame_number in range(diverging_start_frame, end_frame + 1):
        frame_path = os.path.join(target_tif_dir, f'man_track{frame_number:04d}.tif')
        if os.path.exists(frame_path):
            frame = tiff.imread(frame_path)
            frame[frame == to_be_split_id] = new_id
            tiff.imwrite(frame_path, frame)
            print(f"Diverge frame {frame_number}: track {to_be_split_id} -> {new_id}")
    
    # Create a new row for the new track
    new_track_row = pd.DataFrame({
        'Track_ID': [new_id],
        'Start': [diverging_start_frame],
        'End': [end_frame],
        'Parent': [to_be_split_id]
    })
    
    # Append the new track row to the DataFrame 
    new_track_info = track_info._append(new_track_row, ignore_index=True)
    temp = new_track_info['Track_ID'].values
    print(f"Added {new_id} in the track info so now {new_id} is in new_track_info {new_id in temp}")

    # Update the end frame of the original track
    new_track_info.loc[new_track_info['Track_ID'] == to_be_split_id, 'End'] = diverging_start_frame - 1

    new_track_info.to_csv(track_info_file, sep=' ', index=False, header=False)

    return new_track_info

def merge_track(track_info_file, target_tif_dir, to_be_merged_ids):

    track_info = pd.read_csv(track_info_file, sep='\s+', names=['Track_ID', 'Start', 'End', 'Parent'])
    new_track_info = track_info.copy()

    # Determine alpha and beta tracks based on start frames
    alpha_id, beta_id = to_be_merged_ids if track_info.loc[track_info['Track_ID'] == to_be_merged_ids[0], 'Start'].values[0] < track_info.loc[track_info['Track_ID'] == to_be_merged_ids[1], 'Start'].values[0] else to_be_merged_ids[::-1]

    alpha_id, alpha_start_frame, alpha_end_frame, alpha_parent_id = track_info.loc[track_info['Track_ID'] == alpha_id].values[0]
    beta_id, beta_start_frame, beta_end_frame, beta_parent_id = track_info.loc[track_info['Track_ID'] == beta_id].values[0]

    # For frames from alpha_end_frame + 1 to beta_end_frame, all beta track objects are relabeled to alpha.
    sizes_after_alpha_end = []
    for frame_number in range(alpha_end_frame+1, beta_end_frame+1):
        frame_path = os.path.join(target_tif_dir, f'man_track{frame_number:04d}.tif')
        if os.path.exists(frame_path):
            frame = tiff.imread(frame_path)
            beta_mask = (frame == beta_id).astype(np.uint8)
            sizes_after_alpha_end.append(np.sum(beta_mask))
            frame[beta_mask > 0] = alpha_id  # Relabel beta to alpha
            tiff.imwrite(frame_path, frame)
            print(f"Merge frame {frame_number}: track {beta_id} -> {alpha_id}")

    median = np.median(sizes_after_alpha_end)
    mad = np.median(np.abs(sizes_after_alpha_end - median)) 
    lower_bound = median - 2 * mad

    # assume the beta starts at the same frame but ends at (include) alpha_end_frame
    new_beta_start_frame = beta_start_frame.copy()
    new_beta_end_frame = alpha_end_frame.copy()

    if beta_start_frame < alpha_end_frame+1:

        for frame_number in range(beta_start_frame, alpha_end_frame + 1):
            frame_path = os.path.join(target_tif_dir, f'man_track{frame_number:04d}.tif')
            if os.path.exists(frame_path):
                frame = tiff.imread(frame_path)
                beta_mask = (frame == beta_id).astype(np.uint8)
                size_beta = np.sum(beta_mask)
                if size_beta < lower_bound:
                    frame[beta_mask > 0] = 0  # Remove small objects
                    print(f"Removed small object in frame {frame_number} with size {size_beta}")
                else:
                    frame[beta_mask > 0] = alpha_id  # Merge beta into alpha
                    print(f"Merged frame {frame_number}: track {beta_id} -> {alpha_id} (This is before alpha end frame)")
                new_beta_start_frame = frame_number + 1
                tiff.imwrite(frame_path, frame)

    # Update track_info DataFrame
    new_track_info.loc[new_track_info['Track_ID'] == alpha_id, 'End'] = max(alpha_end_frame, beta_end_frame)
    if new_beta_start_frame > new_beta_end_frame: 
        print(f"Remove track ID: {beta_id}")
        new_track_info = new_track_info[new_track_info['Track_ID'] != beta_id]

    else:
        print(f"Putting in new start and end frames {new_beta_start_frame}, {new_beta_end_frame}")
        new_track_info.loc[new_track_info['Track_ID'] == beta_id, 'Start'] = new_beta_start_frame
        new_track_info.loc[new_track_info['Track_ID'] == beta_id, 'End'] = new_beta_end_frame

    new_track_info.to_csv(track_info_file, sep=' ', index=False, header=False)

    temp = new_track_info["Track_ID"].values
    print(f"confirmation that the new track info file do not have beta id {beta_id} {beta_id not in temp}")

    return new_track_info

import numpy as np
from scipy.spatial.distance import cdist

def maj_object_within_radius(frame, point, radius):
    """
    Check if there are any object pixels within a radius around a specified point
    and return the label of the object with the most pixels inside the radius.

    Parameters:
    - frame: numpy array, the pixel assignment matrix.
    - point: tuple (x, y), the coordinates of the given point.
    - radius: float, the radius within which to check for object pixels.

    Returns:
    - int, label of the object with the most pixels within the radius, or 0 if none.
    """
    # Get the coordinates and labels of all non-background pixels
    object_coords = np.column_stack(np.where(frame != 0))
    object_labels = frame[frame != 0]

    # Calculate the Euclidean distance from the given point to each object pixel
    distances = cdist([point], object_coords, metric='euclidean')

    # Get pixels within the specified radius
    within_radius = distances[0] <= radius
    if np.any(within_radius):
        # Count the number of pixels for each label within the radius
        labels_within_radius = object_labels[within_radius]
        unique_labels, counts = np.unique(labels_within_radius, return_counts=True)
        non_zero_labels = unique_labels[unique_labels != 0]
        if len(non_zero_labels) > 0:
            max_count_label = non_zero_labels[np.argmax(counts[unique_labels != 0])]
            return max_count_label
    return 0

#############################################################
# Detect immobile objects
#############################################################

def calculate_movement_and_size(tif_directory, track_info):
    """
    Calculate the movement and size of each track across frames.
    """
    track_movements = {}
    track_sizes = {}
    
    for index, row in track_info.iterrows():
        track_id = row['Track_ID']
        start_frame = row['Start']
        end_frame = row['End']
        
        movement = []
        size = []

        for frame_num in range(start_frame, end_frame + 1):
            frame_file = os.path.join(tif_directory, f"man_track{frame_num:04d}.tif")
            if os.path.exists(frame_file):
                frame = tiff.imread(frame_file)
                
                # Assuming that the track_id is represented by specific pixel values
                track_pixels = np.argwhere(frame == track_id)
                if len(track_pixels) > 0:
                    centroid = np.mean(track_pixels, axis=0)
                    size.append(len(track_pixels))
                    
                    if frame_num > start_frame:
                        movement.append(np.linalg.norm(centroid - prev_centroid))
                    
                    prev_centroid = centroid
        # calc the median and the total dist travelled excluding outliers
        filtered_movement = movement.copy()
        if filtered_movement and len(filtered_movement) > 5: 
            q1, q3 = np.percentile(filtered_movement, [25, 75])
            upper_bound = q3 + 1.5 * (q3-q1)
            filtered_movement = [m for m in filtered_movement if m < upper_bound]


        track_movements[track_id] = np.median(filtered_movement) if len(filtered_movement)>0 else 0
        track_sizes[track_id] = np.median(size) if size else 0

        # print(f"The track {track_id} has movement: {np.median(filtered_movement) if len(filtered_movement)>0 else 0} with size {np.median(size) if size else 0}")

    return track_movements, track_sizes

def find_small_non_moving_tracks(track_info_file, tif_directory, movement_threshold=2.0, size_threshold=173):
    """
    Find tracks that have not moved significantly and are small in size.
    
    Parameters:
    - track_info_file: str, path to the track info text file
    - tif_directory: str, path to the directory containing TIFF files
    - movement_threshold: float, threshold for movement to consider a track as non-moving
    - size_threshold: int, threshold for size to consider a track as small
    
    Returns:
    - list of track IDs that have not moved significantly and are small in size
    """
    track_info = pd.read_csv(track_info_file, sep='\s+', names=['Track_ID', 'Start', 'End', 'Parent'])
    track_movements, track_sizes = calculate_movement_and_size(tif_directory, track_info)
    
    small_non_moving_tracks = [
        track_id for track_id in track_movements
        if track_movements[track_id] <= movement_threshold #and track_sizes[track_id] <= size_threshold
    ]
    
    return sorted(small_non_moving_tracks)

