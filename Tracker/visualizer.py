import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import tifffile as tiff

# Function to create a video from the saved PNGs
def create_video(output_dir, output_video, num_frames, width, height, fps=5, n_img = 2, frames_to_process = None):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width*n_img+10*(n_img-1), height))

    for frame_idx in range(num_frames):

        # Check frame range
        if frames_to_process is not None: 
            if frame_idx not in frames_to_process:
                continue

        if not os.path.isfile(os.path.join(output_dir, f'man_track{frame_idx:04d}.png')): 
            raise ValueError(f"This file does not exists {os.path.join(output_dir, f'man_track{frame_idx:04d}.png')}")

        frame_path = os.path.join(output_dir, f'man_track{frame_idx:04d}.png')
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()
    #print(f'Video saved as {output_video}')

import os
import cv2
import numpy as np
import tifffile as tiff

def process_frames(imgs, tracking_dir, output_dir, frames_to_process = None):
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Load images if `imgs` is a file path
    if os.path.isfile(imgs):
        imgs = np.load(imgs)
        #print(f"Loaded images from {imgs}")

    tif_files = [file for file in os.listdir(tracking_dir) if not file.startswith("._") and file.endswith("tif")]
    tif_files = sorted(tif_files)
    #print(f"Found {len(tif_files)} tif files")

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        #print(f"Created output directory: {output_dir}")

    for i, file in enumerate(tif_files):
        #print(f"Processing file {i}: {file}")

        # Check frame range
        if frames_to_process is not None: 
            if i not in frames_to_process:
                continue
            
        #print(f"Frame {i} is within the specified range")

        # Read the tiff file
        tif = tiff.imread(os.path.join(tracking_dir, file))
        if tif is None:
            print(f"Error reading label image: {os.path.join(tracking_dir, file)}")
            continue

        # Process the corresponding image
        img = imgs[i]
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)  # Normalize the image to 8-bit range
        img = img.astype(np.uint8)  # Convert to 8-bit for visualization
        original_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        annotated_img = original_img.copy()

        unique_labels = np.unique(tif)
        #print(f"Unique labels found: {unique_labels}")

        for label in unique_labels:
            if label == 0:  # Skip the background
                continue

            # Create a mask for the current label
            mask = np.zeros(tif.shape, dtype=np.uint8)
            mask[tif == label] = 255

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #print(f"Found {len(contours)} contours for label {label}")

            # Draw contours and label
            for contour in contours:
                cv2.drawContours(annotated_img, [contour], -1, (57, 255, 20), 1)  # Neon green color with thinner trace
                # Get the bounding box for placing the label
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(annotated_img, str(label), (x, y - 10), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)  # Purple

        # Create a white space (column) between the images
        white_space = np.ones((original_img.shape[0], 10, 3), dtype=np.uint8) * 255

        # Combine the original and annotated images with white space in between
        combined_img = cv2.hconcat([original_img, white_space, annotated_img])
        #print(f"Combined image shape: {combined_img.shape}")

        output_path = os.path.join(output_dir, file.replace(".tif", ".png"))
        # Save the output image
        # print(f"Save and complete file {i}: {file}")
        if not cv2.imwrite(output_path, combined_img):
            print(f"Error saving image: {output_path}")
        #else:
            # print(f"Saved image for frame: {i} at {output_path}")


import os
import numpy as np
import cv2
import tifffile as tiff
import matplotlib.pyplot as plt

# Predefined colors with names
color_map = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'yellow': (255, 255, 0),
    'orange': (255, 165, 0),
    'purple': (128, 0, 128),
    'pink': (255, 192, 203),
    'lime': (0, 255, 0)
}

def get_color_name(index):
    """
    Get the color name and RGB values based on the index.

    Parameters:
    - index: int, index of the color in the color map.

    Returns:
    - tuple: (color_name, color_rgb), where color_name is the name of the color and color_rgb is the RGB tuple.
    """
    color_names = list(color_map.keys())
    color_name = color_names[index % len(color_map)]
    color_rgb = color_map[color_name]
    return color_name, color_rgb

def display_colored_labels(path, labels):
    """
    Display an image with each label colored uniquely.

    Parameters:
    - path: str, path to the directory containing TIFF files or a single TIFF file.
    - labels: list of int, list of labels to be colored.

    Output:
    - Display the image with colored labels.
    """
    # Create a blank canvas for the final image
    final_image = None

    def process_tif_file(file_path):
        nonlocal final_image
        tif = tiff.imread(file_path)

        # Initialize the final image if it hasn't been already
        if final_image is None:
            final_image = np.zeros((tif.shape[0], tif.shape[1], 3), dtype=np.uint8)

        # Color each label with a unique color
        for i, label in enumerate(labels):
            color_name, color_rgb = get_color_name(i)
            # (f"Label {label} is colored with {color_name} (RGB: {color_rgb})")
            mask = (tif == label)
            final_image[mask] = color_rgb

    # Check if the path is a directory or a single file
    if os.path.isdir(path):
        # Process each TIFF file in the directory
        for file in os.listdir(path):
            if file.endswith(".tif"):
                file_path = os.path.join(path, file)
                process_tif_file(file_path)
    else:
        # Process the single TIFF file
        process_tif_file(path)

    # Display the final image
    if final_image is not None:
        plt.figure(figsize=(10, 10))
        plt.title("Colored Labels")
        plt.imshow(final_image)
        plt.axis('off')
        plt.show()
    else:
        print("No TIFF files found or processed.")




