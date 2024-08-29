import tifffile as tiff
import numpy as np
import tifffile as tiff
from skimage.measure import label, regionprops
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from PIL import Image

def load_resize_combine_display(tif_path, png_path, output_path=None):
    """
    Load a TIFF image and a PNG image, resize the PNG to match the TIFF dimensions, and combine/display both images.

    Parameters:
    - tif_path: str, path to the TIFF image file.
    - png_path: str, path to the PNG image file.
    - output_path: str, path to save the combined image (optional).
    """
    # Load the TIFF file
    tif_image = tiff.imread(tif_path)
    
    # Load the PNG file
    png_image = Image.open(png_path)
    
    # Resize the PNG file to match the TIFF dimensions
    png_resized = png_image.resize((tif_image.shape[1], tif_image.shape[0]), Image.LANCZOS)
    
    # Convert the resized PNG image to a numpy array
    png_array = np.array(png_resized)
    
    # Create a figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display the TIFF image with viridis colormap
    ax[0].imshow(tif_image, cmap='viridis')
    ax[0].set_title('TIFF Image (Viridis)')
    ax[0].axis('off')
    
    # Display the resized PNG image
    ax[1].imshow(png_array)
    ax[1].set_title('Resized PNG Image')
    ax[1].axis('off')
    
    plt.tight_layout()
    
    # Save the combined image if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300)
    
    # Show the combined image
    #plt.show()
    plt.close(fig)


def find_highest_object(frame):
    """
    Find the object that has the highest y value within the bottom quarter of the image.

    Parameters:
    - frame: numpy array, the pixel matrix where entries are object assignments.

    Returns:
    - highest_object_id: int, the ID of the object that is closest to the bottom and within the bottom quarter of the image.
    - highest_y: int, the highest y-coordinate of the object.
    """
    # Calculate the threshold for the bottom quarter
    threshold = 2.5 * frame.shape[0] // 4
    
    # Label the objects in the frame
    unique_labels = np.unique(frame)
    
    # Initialize variables to keep track of the highest object
    highest_object_id = 0
    highest_y = 0
    
    # Iterate through each labeled region
    for label in unique_labels:
        if label == 0: 
            continue 
        #print(f"iterating through the find highest y value fucntion and the current rehion is {label}")

        binary_mask = (frame == label).astype(np.uint8)
        region = regionprops(binary_mask)[0]
        # Get the coordinates of the region
        coords = region.coords
        # Find the maximum y-coordinate of the region
        max_y = coords[:, 0].max()
        #print(f"The max y coor is {max_y} and the threshold is {threshold} with the current highest is {highest_y} for track id: {label}")
        # Check if the maximum y-coordinate is within the bottom quarter
        if max_y >= threshold and max_y > highest_y:
            highest_y = max_y
            highest_object_id = label
    
    return highest_object_id, highest_y

def numerical_sort(value):
    """
    Extracts the numeric part from the filename for sorting.
    Assumes that the filename format is '<number>_htert_Run'.
    """
    parts = re.findall(r'\d+', value)
    return int(parts[0]) if parts else value


def link_track_to_well(logfile_directory, tracked_tif_directory, linkfile_directory, output_directory=None):
    linkfile = pd.read_csv(linkfile_directory)
    linkfile = linkfile.sort_values(by='tifs').reset_index(drop=True)
    logfile = pd.read_csv(logfile_directory, dtype={"file_name": str, "R": int, "C": int, "Number_of_droplets": int, "Model_output0": float, "Model_output1": float, "Model_output2": float, "Prediction": int})
    gr_logfile = logfile.groupby(['R', 'C'])

    final_dict = {"track_ID": [], 
                  "after_dispense_frame": [], 
                  "last_tracked_frame": [],
                  "last_tracked_tif": [],
                  "row": [],
                  "col": []}

    for (r, c), group in gr_logfile:
        # Establish the values to be inputted into the dataframe
        track_ID = 0  # Default value meaning no track associated
        after_dispense_frame = sorted(group['file_name'].tolist(), key=numerical_sort, reverse=True)[0]  # This is the last frame for this (r, c) well
        last_tracked_frame = after_dispense_frame 
        last_tracked_tif = ""  # Empty because the image may not have a corresponding tif file

        # Obtain the 3 frames right before the frame after dispense 
        tif_to_consider = []
        img_to_consider = []
        frames_before = [f for f in linkfile["imgs"] if numerical_sort(f) <= numerical_sort(after_dispense_frame)]

        if len(frames_before) > 0: 
            prev_closest_frame = sorted(frames_before, key=numerical_sort, reverse=True)[0]
            tif_after_dispense = linkfile.loc[linkfile["imgs"] == prev_closest_frame]["tifs"]

            if len(tif_after_dispense) != 0:
                i = tif_after_dispense.index[0]
                if i >= 4 and numerical_sort(after_dispense_frame) > numerical_sort(prev_closest_frame): 
                    tif_to_consider = sorted(linkfile.iloc[i-4:i+1]['tifs'], key=numerical_sort, reverse=True)
                    img_to_consider = sorted(linkfile.iloc[i-4:i+1]['imgs'], key=numerical_sort, reverse=True)
                elif i >= 4: 
                    tif_to_consider = sorted(linkfile.iloc[i-4:i]['tifs'], key=numerical_sort, reverse=True)
                    img_to_consider = sorted(linkfile.iloc[i-4:i]['imgs'], key=numerical_sort, reverse=True)
                elif i != 0: 
                    tif_to_consider = sorted(linkfile.iloc[:i]['tifs'], key=numerical_sort, reverse=True)
                    img_to_consider = sorted(linkfile.iloc[:i]['imgs'], key=numerical_sort, reverse=True)

            for tif_name, img_name in zip(tif_to_consider, img_to_consider):
                tracked_tif = tiff.imread(os.path.join(tracked_tif_directory, tif_name))
                track_ID, _ = find_highest_object(tracked_tif)
                last_tracked_frame = img_name
                last_tracked_tif = tif_name
                if track_ID:
                    break  # Stop if we find one that fits into our threshold

        # Append the results to the final dictionary
        final_dict["track_ID"].append(int(track_ID))
        final_dict["after_dispense_frame"].append(after_dispense_frame)
        final_dict["last_tracked_frame"].append(last_tracked_frame)
        final_dict["last_tracked_tif"].append(last_tracked_tif)
        final_dict["row"].append(r)
        final_dict["col"].append(c)

    if output_directory: 
        pd.DataFrame(final_dict).to_csv(os.path.join(output_directory, "track_to_well_unfiltered.csv"), index=False)
    
    return pd.DataFrame(final_dict)

#############################################################
# Section 1: Input Parameters and Configuration
#############################################################
dataset = "A138856A"
printrun = "10dropRun4"

#############################################################
# Section 1: Run Track-to-Well
#############################################################
logfile_directory = f'/projects/steiflab/archive/data/imaging/{dataset}/NozzleImages/{printrun}/LogFile.csv'
tracked_tif_directory = f'/projects/steiflab/scratch/leli/trackastra/{dataset}/{printrun}/tracked_again_postprocessed_2.0'
fluro_directory = f'/projects/steiflab/archive/data/imaging/{dataset}/MicroscopeImages/S0000/'
linkfile_directory = f'/projects/steiflab/scratch/leli/trackastra/{dataset}/{printrun}/tracked_again_postprocessed_2.0/tif_to_img.csv'

df = link_track_to_well(logfile_directory, tracked_tif_directory, linkfile_directory, output_directory = f'/projects/steiflab/scratch/leli/{dataset}/{printrun}/track_to_well')

#############################################################
# Section 1: Postprocess Tracks
#############################################################

def process_track_ids_by_distance(df, track_info, output_directory=None):
    # Process each unique track ID with multiple rows
    track_id_counts = df['track_ID'].value_counts()
    multiple_row_track_ids = track_id_counts[track_id_counts > 1].index

    for track_id in multiple_row_track_ids:
        if track_id <= 0: 
            continue

        track_df = df[df['track_ID'] == track_id]

        # Find the corresponding track info row
        track_info_row = track_info[track_info['Track_ID'] == track_id]
        if track_info_row.empty:
            print(f"Track ID {track_id}: No matching track info found.")
            continue

        # Calculate the distance to the end value for each row
        end_value = track_info_row['End'].values[0]
        df.loc[track_df.index, 'distance_to_end'] = [abs(numerical_sort(i) - end_value) for i in track_df['last_tracked_tif']]

        # Find the row with the minimum distance to the end
        closest_row_idx = df.loc[track_df.index, 'distance_to_end'].idxmin()

        # Label all other rows with -1 in the track_ID column
        df.loc[track_df.index.difference([closest_row_idx]), 'track_ID'] = -1

    # Save the final DataFrame if an output directory is provided
    if output_directory: 
        if "distance_to_end" in list(df.columns): df.drop(columns=['distance_to_end'], inplace=True)
        df.to_csv(os.path.join(output_directory, "track_to_well_pp.csv"), index=False)

    return df

df = pd.read_csv(f"/projects/steiflab/scratch/leli/{dataset}/{printrun}/track_to_well/track_to_well_unfiltered.csv")
new_track_info = pd.read_csv(f"/projects/steiflab/scratch/leli/trackastra/{dataset}/{printrun}/tracked_again_postprocessed_2.0/man_track.txt", sep='\s+', names=['Track_ID', 'Start', 'End', 'Parent'])
processed_df = process_track_ids_by_distance(df, new_track_info, output_directory=f'/projects/steiflab/scratch/leli/{dataset}/{printrun}/track_to_well')

#############################################################
# Section 1: Sanity Check for duplicate tracks
#############################################################
def display_unique_track_ids(df):
    # Filter out non-zero track IDs
    positive_nonzero_track_ids = df[df['track_ID'] > 0]['track_ID']

    # Get the value counts of these track IDs
    track_id_counts = positive_nonzero_track_ids.value_counts()

    # Display the track IDs and their counts
    print("Unique positive non-zero track IDs and their counts:")
    print(track_id_counts)

    # Verify that each track ID has only one row
    duplicate_tracks = track_id_counts[track_id_counts > 1]
    if duplicate_tracks.empty:
        print("All positive non-zero track IDs have only one row.")
    else:
        print("Some track IDs have duplicates:")
        print(duplicate_tracks)

# Assuming df is your DataFrame
display_unique_track_ids(processed_df)