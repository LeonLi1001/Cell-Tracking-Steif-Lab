import os
import numpy as np
import pandas as pd
import cv2
from skimage import measure
from scipy.spatial import distance
from scipy.ndimage import center_of_mass
import tifffile as tiff

def calculate_direction_change(frames, track_id):
    coords = [center_of_mass(frame == track_id) for frame in frames]
    
    # If there's only one frame, return default values
    if len(coords) <= 1:
        return np.pi, np.pi

    deltas = np.diff(coords, axis=0)

    # Ensure deltas is at least 2-dimensional for proper indexing
    if deltas.ndim == 1:
        deltas = deltas.reshape(-1, 2)

    directions = np.arctan2(deltas[:, 1], deltas[:, 0])
    avg_direction = np.mean(directions) if len(directions) > 0 else np.pi
    total_direction = np.arctan2(coords[-1][1] - coords[0][1], coords[-1][0] - coords[0][0])
    
    return avg_direction, total_direction

def calculate_features(track_info_file, tif_directory, track_ids):
    # Load track information
    track_info = pd.read_csv(track_info_file, sep='\s+', header=None, names=['Track_ID', 'Start', 'End', 'Parent'], dtype={'Track_ID': int, 'Start': int, 'End': int, 'Parent': int})
    
    # Initialize a dictionary to store the features
    features_dict = {
        'Track_ID': [],
        'num_frames': [],
        'parent_dist': [],
        'size_avg': [],
        'size_std': [],
        'size_diff': [],
        'size_max': [],
        'size_change': [],
        'shape_change': [],
        'num_obj_nearby_parent': [],
        'num_obj_nearby': [],
        'closest_dist': [],
        'avg_parent_delta_direction': [],
        'total_parent_delta_direction': [],
        'avg_delta_direction': [],
        'total_delta_direction': [],
        'is_parent': [],
        'has_grandparent': [],
        'siblings_count': [],
        'avg_sibling_dist': [],
        'sibling_dist_diff': [],
        'num_parent_frames': [],
        'first_frame_y': [],
    }
    
    for i in track_ids:

        track_id, start_frame, end_frame, parent_id = track_info.loc[track_info['Track_ID'] == i].values[0]

        #print(track_id, start_frame, end_frame, parent_id)
        # Load the frames for the current track
        track_frames = [tiff.imread(os.path.join(tif_directory, f'man_track{frame:04d}.tif')) for frame in range(start_frame, end_frame + 1)]
        
        # Calculate the number of frames
        num_frames = len(track_frames)
        
        # Calculate the size (number of pixels) for each frame and then get the average and std
        sizes = [np.sum(frame == track_id) for frame in track_frames]
        size_avg = np.mean(sizes)
        size_std = np.std(sizes) if num_frames > 1 else 0
        size_max = np.max(sizes)
        size_diff = abs(np.max(sizes) - np.min(sizes))

        # Calculate the size change ratio
        if parent_id != 0:
            #print(f"parent id is: {parent_id}")
            parent_start_frame = track_info[track_info['Track_ID'] == parent_id]['Start'].values[0]
            parent_end_frame = track_info[track_info['Track_ID'] == parent_id]['End'].values[0]
            #print(f"parent end frame is: {parent_end_frame}")
            parent_frame = tiff.imread(os.path.join(tif_directory, f'man_track{parent_end_frame:04d}.tif'))
            parent_size = np.sum(parent_frame == parent_id)
            size_change = size_avg / parent_size if parent_size != 0 else 0
        else:
            size_change = np.nan
        
        # Calculate the shape change using Hu moments
        contours, _ = cv2.findContours((track_frames[0] == track_id).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hu_moments = cv2.HuMoments(cv2.moments(contours[0])).flatten() if contours else np.zeros(7)
        
        if parent_id != 0:
            parent_contours, _ = cv2.findContours((parent_frame == parent_id).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            parent_hu_moments = cv2.HuMoments(cv2.moments(parent_contours[0])).flatten() if parent_contours else np.zeros(7)
            shape_change = np.sum(np.abs(hu_moments - parent_hu_moments))
        else:
            shape_change = np.nan
        
        # Calculate the number of objects nearby the parent and current track's first frame
        radius = 0.15 * max(track_frames[0].shape)
        
        def count_nearby_objects(frame, track_id):
            centroids = [center_of_mass(frame == obj_id) for obj_id in np.unique(frame) if obj_id != track_id and obj_id != 0]
            track_centroid = center_of_mass(frame == track_id)
            nearby_objects = [centroid for centroid in centroids if distance.euclidean(centroid, track_centroid) <= radius]
            return len(nearby_objects)
        
        if parent_id != 0:
            num_obj_nearby_parent = count_nearby_objects(parent_frame, parent_id)
        else:
            num_obj_nearby_parent = 0
        
        num_obj_nearby = count_nearby_objects(track_frames[0], track_id)
        
        # Calculate the Euclidean distance to the nearest other track
        all_other_tracks = np.unique(track_frames[0])
        if len(all_other_tracks) > 1:
            distances = [distance.euclidean(center_of_mass(track_frames[0] == other_id), center_of_mass(track_frames[0] == track_id))
                         for other_id in all_other_tracks if other_id != track_id]
            closest_dist = min(distances)
        else:
            closest_dist = 0
        
        if parent_id != 0:
            parent_frames = [tiff.imread(os.path.join(tif_directory, f'man_track{frame:04d}.tif')) for frame in range(parent_start_frame, parent_end_frame + 1)]
            # print(f"this is the size of the parent frames: {len(parent_frames)}, from {parent_start_frame} to {parent_end_frame}")
            avg_parent_delta_direction, total_parent_delta_direction = calculate_direction_change(parent_frames, parent_id)
        else:
            avg_parent_delta_direction = total_parent_delta_direction = np.pi
        
        avg_delta_direction, total_delta_direction = calculate_direction_change(track_frames, track_id)
        
        # Check if the current track is a parent
        is_parent = int(track_info['Parent'].isin([track_id]).any())
        
        # Check if the current track has a grandparent
        has_grandparent = int(parent_id != 0 and track_info.loc[track_info['Track_ID'] == parent_id, 'Parent'].values[0] != 0)
        
        # Count the number of siblings (tracks sharing the same parent)
        siblings = track_info[track_info['Parent'] == parent_id]
        siblings_count = track_info[track_info['Parent'] == parent_id].shape[0] if parent_id != 0 else 0

        # Calculate the average, and difference between max and min distance to siblings
        if siblings_count > 1:
            sibling_distances = []
            for frame in track_frames:
                for sibling_id in siblings['Track_ID']:
                    if sibling_id != track_id and sibling_id in frame:
                        sibling_centroid = center_of_mass(frame == sibling_id)
                        track_centroid = center_of_mass(frame == track_id)
                        # print(f"The track {i} has sibling centroid {sibling_centroid} and the track centroid {track_centroid} when in this frame there are tracks {np.unique(frame)}")
                        sibling_distances.append(distance.euclidean(sibling_centroid, track_centroid))
            avg_sibling_dist = np.mean(sibling_distances) if sibling_distances else 0
            sibling_dist_diff = (np.max(sibling_distances) - np.min(sibling_distances)) if sibling_distances else 0
        else:
            avg_sibling_dist = sibling_dist_diff = 0
        
        # Calculate the number of parent frames
        num_parent_frames = len(parent_frames) if parent_id != 0 else 0
        
        # Get the y position of the first frame
        first_frame_y = center_of_mass(track_frames[0] == track_id)[0]
        
        # Append the features to the dictionary
        features_dict['Track_ID'].append(track_id)
        features_dict['num_frames'].append(num_frames)
        features_dict['parent_dist'].append(closest_dist)
        features_dict['size_avg'].append(size_avg)
        features_dict['size_std'].append(size_std)
        features_dict['size_max'].append(size_max)
        features_dict['size_diff'].append(size_diff)
        features_dict['size_change'].append(size_change)
        features_dict['shape_change'].append(shape_change)
        features_dict['num_obj_nearby_parent'].append(num_obj_nearby_parent)
        features_dict['num_obj_nearby'].append(num_obj_nearby)
        features_dict['closest_dist'].append(closest_dist)
        features_dict['avg_parent_delta_direction'].append(avg_parent_delta_direction)
        features_dict['total_parent_delta_direction'].append(total_parent_delta_direction)
        features_dict['avg_delta_direction'].append(avg_delta_direction)
        features_dict['total_delta_direction'].append(total_delta_direction)
        features_dict['is_parent'].append(is_parent)
        features_dict['has_grandparent'].append(has_grandparent)
        features_dict['siblings_count'].append(siblings_count)
        features_dict['avg_sibling_dist'].append(avg_sibling_dist)
        features_dict['sibling_dist_diff'].append(sibling_dist_diff)
        features_dict['num_parent_frames'].append(num_parent_frames)
        features_dict['first_frame_y'].append(first_frame_y)

    # print(f"features_dict is {features_dict}")
    return features_dict
