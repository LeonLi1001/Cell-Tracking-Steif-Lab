import os
import numpy as np
import pandas as pd
import cv2
from skimage import measure
from scipy.spatial import distance
from scipy.ndimage import center_of_mass
import tifffile as tiff
import numpy as np
from skimage.measure import regionprops
from skimage import img_as_ubyte
from PIL import Image
import matplotlib.pyplot as plt
import diplib as dip
import sys
from skimage.measure import label as sk_label
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def get_diameter_cellenONE(frame, track_id, um_per_pixel = 0.69):
    area = np.sum(frame == track_id)
    return 2 * np.sqrt(area / np.pi) * um_per_pixel

def get_elongation_cellenONE(frame, track_id):
    # Create a binary mask for the given track ID
    binary_mask = (frame == track_id).astype(np.uint8)
    
    # Label the binary mask
    labeled_mask = sk_label(binary_mask)
    
    # Extract region properties
    regions = regionprops(labeled_mask)
    
    if len(regions) == 0:
        raise ValueError(f"No object with track ID {track_id} found in the frame.")
    
    region = regions[0]
    
    if region.minor_axis_length == 0: 
        print("The region minor axis lnegth is zero, we retrun zero to avoid zero division error")
        return 0

    
    return region.major_axis_length / region.minor_axis_length

def get_gray_intensity_cellenONE(frame, intensity_image, track_id):
    """
    Calculate the mean normalized intensity of the object with the given track ID in the frame.

    Parameters:
    - frame: numpy array, the pixel matrix where entries are object assignments.
    - intensity_image: numpy array, the intensity values of the corresponding pixels.
    - track_id: int, the ID of the track for which to calculate the intensity.

    Returns:
    - mean_intensity: float, the mean normalized intensity of the object.
    """
    # Create a binary mask for the given track ID
    binary_mask = (frame == track_id).astype(np.uint8)
    
    # Label the binary mask
    labeled_mask = sk_label(binary_mask)
    
    # Extract region properties
    regions = regionprops(labeled_mask, intensity_image=intensity_image)
    
    if len(regions) == 0:
        raise ValueError(f"No object with track ID {track_id} found in the frame.")
    
    region = regions[0]
    
    # Get the intensity values of the pixels inside the object
    intensity_values = region.intensity_image[region.image]
    
    return np.mean(intensity_values)


def get_glcm_features(frame, intensity_image, track_id, get_e):
    """
    Calculate the GLCM features of the object with the given track ID in the frame.

    Parameters:
    - frame: numpy array, the pixel matrix where entries are object assignments.
    - intensity_image: numpy array, the intensity values of the corresponding pixels.
    - track_id: int, the ID of the track for which to calculate the GLCM features.

    Returns:
    - glcm_features: tuple, containing GLCM features like energy and homogeneity.
    """
    # Create a binary mask for the given track ID
    binary_mask = (frame == track_id).astype(np.uint8)
    
    # Label the binary mask
    labeled_mask = sk_label(binary_mask)
    
    # Extract region properties
    regions = regionprops(labeled_mask, intensity_image=intensity_image)
    
    if len(regions) == 0:
        raise ValueError(f"No object with track ID {track_id} found in the frame.")
    
    region = regions[0]
    
    # Get the intensity values of the pixels inside the object
    intensity_values = region.intensity_image[region.image]
    
    # Normalize intensity values to the range [0, 255]
    intensity_values = np.round((intensity_values - np.min(intensity_values)) / (np.max(intensity_values) - np.min(intensity_values)) * 255).astype(np.uint8)
    
    # Reshape intensity values to 2D if they are not already
    if intensity_values.ndim == 1:
        size = int(np.sqrt(intensity_values.size))
        intensity_values = intensity_values[:size*size].reshape((size, size))
    
    # Calculate GLCM
    glcm = graycomatrix(intensity_values, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    
    # Calculate and return GLCM features
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    
    if get_e: return energy
    return homogeneity


def get_w(frame, track_id):
    """
    Converts a binary mask to a bounding box.

    :param numpy.ndarray mask: Binary mask.
    :return: Bounding box in the format (x, y, w, h).
    :rtype: list[int]
    """
    rows, cols = np.where(frame == track_id)
    x1, x2 = np.min(cols), np.max(cols)
    y1, y2 = np.min(rows), np.max(rows)
    return x2-x1

def get_h(frame, track_id):
    """
    Converts a binary mask to a bounding box.

    :param numpy.ndarray mask: Binary mask.
    :return: Bounding box in the format (x, y, w, h).
    :rtype: list[int]
    """
    #print(f'These are the unique values of thje frame: {np.unique(frame)}')

    rows, cols = np.where(frame == track_id)
    x1, x2 = np.min(cols), np.max(cols)
    y1, y2 = np.min(rows), np.max(rows)
    return y2-y1

def get_circ(frame, track_id):
    """
    Calculate the circularity and eccentricity of the object with the given track ID in the frame.

    Parameters:
    - frame: numpy array, the pixel matrix where entries are object assignments.
    - track_id: int, the ID of the track for which to calculate the features.

    Returns:
    - circularity: float, the circularity of the object. It is computed by the ratio StD/Mean of the Radius feature, and is 0 for a perfect circle
    - roundness: float, the eccentricity of the object. This measure is in the range (0,1], with 1 for a perfect circle.
    """
    '''# Create a binary mask for the given track ID
    binary_mask = (frame == track_id).astype(np.uint8)
    
    # Label the binary mask
    labeled_mask = label(binary_mask)
    
    # Extract region properties
    regions = regionprops(labeled_mask)
    
    if len(regions) == 0:
        raise ValueError(f"No object with track ID {track_id} found in the frame.")
    
    region = regions[0]
    
    # Calculate circularity
    area = region.area
    perimeter = region.perimeter
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
    
    # Calculate eccentricity
    eccentricity = region.eccentricity'''

    # Create a binary mask for the given track ID
    binary_mask = (frame == track_id).astype(np.uint8)
    
    # Use DIPlib for advanced measurement
    labels = dip.Label(binary_mask > 0)
    msr = dip.MeasurementTool.Measure(labels, features=["Perimeter", "Size", "Roundness", "Circularity"])

    # Extract measurements
    roundness = msr[1]["Roundness"][0]
    circularity = msr[1]["Circularity"][0]
    
    return roundness

def get_intensity(frame, intensity_image, track_id):
    """
    Calculate the mean normalized intensity of the object with the given track ID in the frame.

    Parameters:
    - frame: numpy array, the pixel matrix where entries are object assignments.
    - intensity_image: numpy array, the intensity values of the corresponding pixels.
    - track_id: int, the ID of the track for which to calculate the intensity.

    Returns:
    - mean_intensity: float, the mean normalized intensity of the object.
    """
    # Create a binary mask for the given track ID
    binary_mask = (frame == track_id).astype(np.uint8)
    
    # Label the binary mask
    labeled_mask = sk_label(binary_mask)
    
    # Extract region properties
    regions = regionprops(labeled_mask, intensity_image=intensity_image)
    
    if len(regions) == 0:
        raise ValueError(f"No object with track ID {track_id} found in the frame.")
    
    region = regions[0]
    
    # Get the intensity values of the pixels inside the object
    intensity_values = region.intensity_image[region.image]
    
    # Calculate the mean and standard deviation for the entire intensity image
    image_mean = np.mean(intensity_image)
    image_std = np.std(intensity_image)
    
    # Normalize the intensity values using Z-score normalization
    normalized_intensity_values = (intensity_values - image_mean) / image_std
    
    # Calculate the mean normalized intensity
    mean_intensity = np.mean(normalized_intensity_values)
    
    return mean_intensity


def linear_plot_with_summary(values, names, title, output_directory=None, xlab = 'Track ID'):
    """
    Plot a linear graph of values with summary statistics in the legend and dotted lines for bounds.

    Parameters:
    - values: list or numpy array of numerical values.
    - names: list of str, names corresponding to each value.
    - title: str, title of the plot.
    - output_directory: str, directory to save the plot.
    """
    values = np.array(values)
    names = np.array(names)
    
    # Calculate summary statistics
    mean_val = np.mean(values)
    median_val = np.median(values)
    min_val = np.min(values)
    max_val = np.max(values)
    std_val = np.std(values)
    q1_val = np.percentile(values, 25)
    q3_val = np.percentile(values, 75)
    iqr_val = q3_val - q1_val
    lower_bound = max(0, q1_val - 1.5 * iqr_val)
    upper_bound = q3_val + 1.5 * iqr_val
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the values
    ax.plot(names, values, marker='o', linestyle='-', color='b', label=title)
    
    # Plot bounds as solid lines
    ax.axhline(lower_bound, color='cyan', linestyle='-', linewidth=2, label=f'Lower Bound: {lower_bound:.2f}')
    ax.axhline(upper_bound, color='magenta', linestyle='-', linewidth=2, label=f'Upper Bound: {upper_bound:.2f}')
    
    # Add summary statistics to the legend
    summary_stats = (f'Mean: {mean_val:.2f}\n'
                     f'Median: {median_val:.2f}\n'
                     f'Std: {std_val:.2f}\n'
                     f'Min: {min_val:.2f}\n'
                     f'Max: {max_val:.2f}')
    ax.plot([], [], ' ', label=summary_stats)
    
    # Highlight points outside the bounds and add their names
    for i, (value, name) in enumerate(zip(values, names)):
        if value < lower_bound or value > upper_bound:
            ax.annotate(name, (name, value), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='red')
    
    # Set the title and labels
    ax.set_title(f"Distribution of Track {title}", fontsize=16, fontweight='bold')
    ax.set_xlabel(xlab, fontsize=14)
    ax.set_ylabel(title, fontsize=14)
    
    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add a grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend
    ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
    
    # Show the plot
    plt.tight_layout()
    if output_directory is not None:
        os.makedirs(output_directory, exist_ok=True)
        plt.savefig(os.path.join(output_directory, f'{title}_linear_plot.png'), dpi=300)
    plt.show()


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def histogram_with_best_fit(values, title, xlab, output_directory=None, dataset = None):
    """
    Plot a histogram with a best-fit line and summary statistics in the legend.
    
    Parameters:
    - values: list or numpy array of numerical values.
    - title: str, title of the plot.
    - output_directory: str, directory to save the plot.
    """
    values = np.array(values)
    
    # Calculate summary statistics
    mean_val = np.mean(values)
    median_val = np.median(values)
    min_val = np.min(values)
    max_val = np.max(values)
    std_val = np.std(values)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the histogram
    count, bins, ignored = plt.hist(values, bins=30, density=False, alpha=0.6, color='b', label=title)
    
    # Plot the best-fit line
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean_val, std_val)
    plt.plot(x, p * len(values) * (bins[1] - bins[0]), 'k', linewidth=2, label='Best Fit Line')
    
    # Add summary statistics to the legend
    summary_stats = (f'Mean: {mean_val:.2f}\n'
                     f'Median: {median_val:.2f}\n'
                     f'Std: {std_val:.2f}\n'
                     f'Min: {min_val:.2f}\n'
                     f'Max: {max_val:.2f}')
    ax.plot([], [], ' ', label=summary_stats)
    
    # Set the title and labels
    ax.set_title(f"Distribution of {dataset} {title}", fontsize=16, fontweight='bold')
    ax.set_xlabel(xlab, fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    
    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add a grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend
    ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
    
    # Show the plot
    plt.tight_layout()
    if output_directory is not None:
        os.makedirs(output_directory, exist_ok=True)
        plt.savefig(os.path.join(output_directory, f'{dataset}_{title}_histogram.png'), dpi=300)
    plt.show()

import matplotlib.pyplot as plt

def create_boxplot(data, xticklabels, title, output_directory=None):

    if len(data) != len(xticklabels):
        raise ValueError (f"Within the boxplot creating function, the length of the input value and the x labels are not the identical: {len(data)}, {len(xticklabels)}")
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create the boxplot
    ax.boxplot(data)
    
    # Set the x-axis labels and ticks
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Datasets")
    ax.set_title(f"Boxplot of {title}", fontsize=16, fontweight='bold')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.tight_layout()
    if output_directory is not None:
        os.makedirs(output_directory, exist_ok=True)
        plt.savefig(os.path.join(output_directory, f'{title}_boxplot.png'), dpi=300)
    plt.show()


def create_violinplot(data, xticklabels, title, output_directory=None):
    """
    Create a violin plot for multiple lists of values.

    Parameters:
    - data: List of lists containing numerical values to plot.
    - xticklabels: Labels for the x-axis corresponding to each list in data.
    - title: Title of the plot.
    - output_directory: Directory to save the plot (default is None).
    """
    if len(data) != len(xticklabels):
        raise ValueError(f"Within the violin plot creating function, the length of the input value and the x labels are not identical: {len(data)}, {len(xticklabels)}")
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create the violin plot
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    
    # Set the x-axis labels and ticks
    ax.set_xticks(range(1, len(xticklabels) + 1))
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Datasets")
    ax.set_title(f"Violin Plot of {title}", fontsize=16, fontweight='bold')
    
    # Customize the plot
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
        vp = parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.tight_layout()
    if output_directory is not None:
        os.makedirs(output_directory, exist_ok=True)
        plt.savefig(os.path.join(output_directory, f'{title}_violinplot.png'), dpi=300)
    plt.show()

def create_violinplot_with_boxplot(data, xticklabels, title, output_directory=None):
    """
    Create a violin plot with embedded boxplots for multiple lists of values.

    Parameters:
    - data: List of lists containing numerical values to plot.
    - xticklabels: Labels for the x-axis corresponding to each list in data.
    - title: Title of the plot.
    - output_directory: Directory to save the plot (default is None).
    """
    if len(data) != len(xticklabels):
        raise ValueError(f"Within the violin plot creating function, the length of the input value and the x labels are not identical: {len(data)}, {len(xticklabels)}")
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create the violin plot
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    
    # Set the x-axis labels and ticks
    #ax.set_xticks(range(1, len(xticklabels) + 1))
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Datasets")
    ax.set_title(f"Violin Plot of {title} across datasets", fontsize=16, fontweight='bold')
    
    # Customize the violin plot
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1)
    
    # Add the boxplot inside the violin plot
    boxprops = dict(linestyle='-', linewidth=1.5, color='black')
    whiskerprops = dict(linestyle='-', linewidth=1.5, color='black')
    capprops = dict(linestyle='-', linewidth=1.5, color='black')
    medianprops = dict(linestyle='-', linewidth=1.5, color='red')
    meanprops = dict(linestyle='--', linewidth=1.5, color='blue')
    
    ax.boxplot(data, positions=np.arange(1, len(data) + 1), widths=0.1,
               boxprops=boxprops, whiskerprops=whiskerprops,
               capprops=capprops, medianprops=medianprops,
               meanprops=meanprops, showmeans=True, patch_artist=True, showfliers=True, flierprops=dict(marker='o', color='red', markersize=4))
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.tight_layout()
    if output_directory is not None:
        os.makedirs(output_directory, exist_ok=True)
        plt.savefig(os.path.join(output_directory, f'{title}_violinplot.png'), dpi=300)
    plt.show()

def feature_extraction_across_tracks(track_info_file, tif_directory, png_directory, features = ['area', 'length', 'width', 'circularity', 'intensity', 'energy', 'homogeneity', 'diameter', 'elongation', 'intensity_not_norm'], track = 'all', track_ids = None, output_directory = None, display_plot = True, dataset = None):

    # Load track information
    track_info = pd.read_csv(track_info_file, sep='\s+', header=None, names=['Track_ID', 'Start', 'End', 'Parent'], dtype={'Track_ID': int, 'Start': int, 'End': int, 'Parent': int})

    if track == 'all':
        track_ids = list(sorted(track_info['Track_ID']))
        if track_ids is None: raise RuntimeError("The track ID is None:(")

    area = [] 
    length = []
    width  = []
    circularity = []
    intensity = []
    energy = []
    homogeneity = []
    diameter = []
    elongation = []
    intensity_not_norm = []
    track_ids_to_save = []

    for i in track_ids:

        track_id, start_frame, end_frame, parent_id = track_info.loc[track_info['Track_ID'] == i].values[0]
        #print(f"the current track ID is: {track_id}")

        # Load the frames for the current track
        track_frames = []
        intensity_frames = []
        for frame in range(start_frame, end_frame+1):
            temp = tiff.imread(os.path.join(tif_directory, f'man_track{frame:04d}.tif'))
            if track_id not in np.unique(temp):
                #print(f"The track id: {track_id} is not in the frame {frame}")
                continue
            track_frames.append(tiff.imread(os.path.join(tif_directory, f'man_track{frame:04d}.tif')))
            intensity_frame = np.array(Image.open(os.path.join(png_directory, f'man_track{frame:04d}.png')).convert('L'))
            w =int((intensity_frame.shape[1]-10)/2)
            intensity_frames.append(intensity_frame[:,:w])

        #track_frames = [tiff.imread(os.path.join(tif_directory, f'man_track{frame:04d}.tif')) for frame in range(start_frame, end_frame + 1)]
        #intensity_frames = [np.array(Image.open(os.path.join(png_directory, f'man_track{frame:04d}.png')).convert('L'))[:,:192] for frame in range(start_frame, end_frame + 1)]
        #print(f"The shape of track is {track_frames[0].shape} and the intensity is {intensity_frames[0].shape}")
        #print(f"There are {len(track_frames)} which are")
        # Calculate the number of frames
        num_frames = len(track_frames)

        if 'area' in features:
            if np.isnan(np.median([np.sum(frame == track_id) for frame in track_frames])): 
                #print(f"The size of the frames is {len(track_frames)} with start and end frames {start_frame, end_frame}")
                #print(f"The track id is {track_id}")
                return None
            area.append(np.median([np.sum(frame == track_id) for frame in track_frames]))
        if 'length' in features:
            length.append(np.median([get_h(frame, track_id) for frame in track_frames]))
        if 'width' in features:
            width.append(np.median([get_w(frame, track_id) for frame in track_frames]))
        if 'circularity' in features: 
            circularity.append(np.median([get_circ(frame, track_id) for frame in track_frames]))
        if 'intensity' in features: 
            intensity.append(np.median([get_intensity(frame = track_frame, intensity_image = intensity_frame, track_id = track_id) for track_frame, intensity_frame in zip(track_frames, intensity_frames)]))
        if 'energy' in features: 
            energy.append(np.median([get_glcm_features(frame = track_frame, intensity_image = intensity_frame, track_id = track_id, get_e = True) for track_frame, intensity_frame in zip(track_frames, intensity_frames)]))
        if 'homogeneity' in features: 
            homogeneity.append(np.median([get_glcm_features(frame = track_frame, intensity_image = intensity_frame, track_id = track_id, get_e = False) for track_frame, intensity_frame in zip(track_frames, intensity_frames)]))
        if 'diameter' in features: 
            diameter.append(np.median([get_diameter_cellenONE(frame, track_id) for frame in track_frames]))
        if 'elongation' in features: 
            elongation.append(np.median([get_elongation_cellenONE(frame, track_id) for frame in track_frames]))
        if 'intensity_not_norm' in features: 
            intensity_not_norm.append(np.median([get_gray_intensity_cellenONE(frame = track_frame, intensity_image = intensity_frame, track_id = track_id) for track_frame, intensity_frame in zip(track_frames, intensity_frames)]))

        track_ids_to_save.append(track_id)

    if display_plot: 
        if 'area' in features:
            #print(f"The area is {area}")
            #print(f"The track ids are {track_ids}")
            histogram_with_best_fit(area, "Histogram of Object Area", 'area', output_directory, dataset = dataset)
        if 'length' in features:
            histogram_with_best_fit(length, "Histogram of Object length", 'length', output_directory, dataset = dataset)
        if 'width' in features:
            histogram_with_best_fit(width, "Histogram of Object width", 'width', output_directory, dataset = dataset)
        if 'circularity' in features: 
            histogram_with_best_fit(circularity, "Histogram of Object circularity", 'circularity', output_directory, dataset = dataset)
        if 'intensity' in features: 
            histogram_with_best_fit(intensity, "Histogram of Object intensity", 'intensity', output_directory, dataset = dataset)
        if 'energy' in features: 
            histogram_with_best_fit(energy, "Histogram of Object energy", 'energy', output_directory, dataset = dataset)
        if 'homogeneity' in features: 
            histogram_with_best_fit(homogeneity, "Histogram of Object homogeneity", 'homogeneity', output_directory, dataset = dataset)
        if 'diameter' in features: 
            histogram_with_best_fit(diameter, "Histogram of Object diameter", 'diameter', output_directory, dataset = dataset)
        if 'elongation' in features: 
            histogram_with_best_fit(elongation, "Histogram of Object elongation", 'elongation', output_directory, dataset = dataset)
        if 'intensity_not_norm' in features: 
            histogram_with_best_fit(intensity_not_norm, "Histogram of Object intensity non-normalized", 'intensity_not_norm', output_directory, dataset = dataset)
    
    return pd.DataFrame({'track_id': track_ids_to_save,
                        'area': area, 
                        'length': length, 
                        'width': width, 
                        'circularity': circularity, 
                        'intensity': intensity,
                        'energy': energy, 
                        'homogeneity': homogeneity,
                        'diameter': diameter, 
                        'elongation': elongation, 
                        'intensity_not_norm': intensity_not_norm, 
                        })

def feature_extraction_across_frames(track_info_file, tif_directory, png_directory, features = ['area', 'length', 'width', 'circularity', 'intensity', 'energy', 'homogeneity', 'diameter', 'elongation', 'intensity_not_norm'], track = 'all', track_ids = None, output_directory = None):

    # Load track information
    track_info = pd.read_csv(track_info_file, sep='\s+', header=None, names=['Track_ID', 'Start', 'End', 'Parent'], dtype={'Track_ID': int, 'Start': int, 'End': int, 'Parent': int})

    if track == 'all':
        track_ids = list(sorted(track_info['Track_ID']))
        if track_ids is None: raise RuntimeError("The track ID is None:(")


    for i in track_ids: 
        area = [] 
        length = []
        width  = []
        circularity = []
        intensity = []
        energy = []
        homogeneity = []
        diameter = []
        elongation = []
        intensity_not_norm = []
        
        track_id, start_frame, end_frame, parent_id = track_info.loc[track_info['Track_ID'] == i].values[0]

        frame_nums = []
        for frame in range(start_frame, end_frame+1):
            track_frame = tiff.imread(os.path.join(tif_directory, f'man_track{frame:04d}.tif'))
            intensity_frame = np.array(Image.open(os.path.join(png_directory, f'man_track{frame:04d}.png')).convert('L'))
            w =int((intensity_frame.shape[1]-10)/2)
            intensity_frames.append(intensity_frame[:,:w])
            #intensity_frame = np.array(Image.open(os.path.join(png_directory, f'man_track{frame:04d}.png')).convert('L'))[:,:192]

            if track_id not in np.unique(track_frame):
                continue
            frame_nums.append(frame)
            #print(f"input {frame} into frame nums {frame_nums}")
            if 'area' in features:
                #print(f"input {np.sum(track_frame == track_id)} into area {area}")
                area.append(np.sum(track_frame == track_id))
            if 'length' in features:
                length.append(get_h(track_frame, track_id))
            if 'width' in features:
                width.append(get_w(track_frame, track_id))
            if 'circularity' in features: 
                circularity.append(get_circ(track_frame, track_id))
            if 'intensity' in features: 
                intensity.append(get_intensity(frame = track_frame, intensity_image = intensity_frame, track_id = track_id))
            if 'energy' in features: 
                energy.append(get_glcm_features(frame = track_frame, intensity_image = intensity_frame, track_id = track_id, get_e = True))
            if 'homogeneity' in features: 
                homogeneity.append(get_glcm_features(frame = track_frame, intensity_image = intensity_frame, track_id = track_id, get_e = False))
            if 'diameter' in features:
                diameter.append(get_diameter_cellenONE(track_frame, track_id))
            if 'elongation' in features:
                elongation.append(get_elongation_cellenONE(track_frame, track_id))
            if 'intensity_not_norm' in features: 
                intensity_not_norm.append(get_gray_intensity_cellenONE(frame = track_frame, intensity_image = intensity_frame, track_id = track_id))
            


        if 'area' in features:
            #print(f"The area is {area} with the frame numbers {frame_nums}")
            linear_plot_with_summary(area, frame_nums, f'track#{track_id} area', xlab = "area", output_directory=output_directory)
        if 'length' in features:
            linear_plot_with_summary(length, frame_nums, f'track#{track_id} length', xlab = "length", output_directory=output_directory)
        if 'width' in features:
            linear_plot_with_summary(width, frame_nums, f'track#{track_id} width', xlab = "width", output_directory=output_directory)
        if 'circularity' in features: 
            linear_plot_with_summary(circularity, frame_nums, f'track#{track_id} circularity', xlab = "circularity", output_directory=output_directory)
        if 'intensity' in features: 
            linear_plot_with_summary(intensity, frame_nums, f'track#{track_id} intensity', xlab = "intensity", output_directory=output_directory)
        if 'energy' in features:
            linear_plot_with_summary(energy, frame_nums, f'track#{track_id} energy', xlab = "energy", output_directory=output_directory)
        if 'homogeneity' in features: 
            linear_plot_with_summary(homogeneity, frame_nums, f'track#{track_id} homogeneity', xlab = "homogeneity", output_directory=output_directory)
        if 'diameter' in features: 
            linear_plot_with_summary(diameter, frame_nums, f'track#{track_id} diameter', xlab = "diameter", output_directory=output_directory)
        if 'elongation' in features: 
            linear_plot_with_summary(elongation, frame_nums, f'track#{track_id} elongation', xlab = "elongation", output_directory=output_directory)
        if 'intensity_not_norm' in features: 
            linear_plot_with_summary(intensity_not_norm, frame_nums, f'track#{track_id} intensity_not_norm', xlab = "intensity_not_norm", output_directory=output_directory)


def feature_extraction_across_dataset(track_info_files, tif_directories, png_directories, dataset_names, features = ['area', 'length', 'width', 'circularity', 'intensity', 'energy', 'homogeneity', 'diameter', 'elongation', 'intensity_not_norm'], track = 'all', track_ids = None, output_directory = None, display_plot = True):
    if len(dataset_names) != len(track_info_files) or len(dataset_names) != len(tif_directories) or len(dataset_names) != len(png_directories):
        print(f"Error: Invalid Input: There are {len(dataset_names)} datasets but there are len(track_info_files) track info files, {len(tif_directories)} tif folders, and {png_directories} image folders.")
        sys.exit(1)
    
    area = [] 
    length = []
    width  = []
    circularity = []
    intensity = []
    energy = []
    homogeneity = []
    diameter = []
    elongation = []
    intensity_not_norm = []

    for i, name in enumerate(dataset_names):
        temp = feature_extraction_across_tracks(track_info_files[i], tif_directories[i], png_directories[i], features = features, output_directory = output_directory, display_plot = False)
        if 'area' in features:
            area.append(temp['area'])
        if 'length' in features:
            length.append(temp['length'])
        if 'width' in features:
            width.append(temp['width'])
        if 'circularity' in features: 
            circularity.append(temp['circularity'])
        if 'intensity' in features: 
            intensity.append(temp['intensity'])
        if 'energy' in features: 
            energy.append(temp['energy'])
        if 'homogeneity' in features: 
            homogeneity.append(temp['homogeneity'])
        if 'diameter' in features: 
            diameter.append(temp['diameter'])
        if 'elongation' in features: 
            elongation.append(temp['elongation'])
        if 'intensity_not_norm' in features: 
            intensity_not_norm.append(temp['intensity_not_norm'])
        


    if 'area' in features:
        create_violinplot_with_boxplot(area, xticklabels = dataset_names, title = 'area', output_directory=output_directory)
    if 'length' in features:
        create_violinplot_with_boxplot(length, xticklabels = dataset_names, title = 'length', output_directory=output_directory)
    if 'width' in features:
        create_violinplot_with_boxplot(width, xticklabels = dataset_names, title = 'width', output_directory=output_directory)
    if 'circularity' in features: 
        create_violinplot_with_boxplot(circularity, xticklabels = dataset_names, title = 'circularity', output_directory=output_directory)
    if 'intensity' in features: 
        create_violinplot_with_boxplot(intensity, xticklabels = dataset_names, title = 'intensity', output_directory=output_directory)
    if 'energy' in features: 
        create_violinplot_with_boxplot(energy, xticklabels = dataset_names, title = 'energy', output_directory=output_directory)
    if 'homogeneity' in features: 
        create_violinplot_with_boxplot(homogeneity, xticklabels = dataset_names, title = 'homogeneity', output_directory=output_directory)
    if 'diameter' in features: 
        create_violinplot_with_boxplot(diameter, xticklabels = dataset_names, title = 'diameter', output_directory=output_directory)
    if 'elongation' in features: 
        create_violinplot_with_boxplot(elongation, xticklabels = dataset_names, title = 'elongation', output_directory=output_directory)
    if 'intensity_not_norm' in features: 
        create_violinplot_with_boxplot(intensity_not_norm, xticklabels = dataset_names, title = 'intensity_not_norm', output_directory=output_directory)


## Example Usage:
'''dataset_names = ["A138974A", "A146237A", "A118880", "A138856A"]

track_info_files = [
    "/projects/steiflab/scratch/leli/trackastra/A138974A/PrintRun_Apr1223_1311/tracked_postprocessed_2.0/man_track_test.txt",
    "/projects/steiflab/scratch/leli/trackastra/A146237A/PrintRun_Mar2824_1524/tracked_postprocessed_2.0/man_track.txt",
    "/projects/steiflab/scratch/leli/trackastra/A118880/PrintRun_Jan2624_1252/tracked_again_postprocessed_2.0/man_track.txt", 
    "/projects/steiflab/scratch/leli/trackastra/A138856A/htert_20230822_131349_843.Run/tracked_postprocessed_2.0/man_track.txt"
]
tif_directories = [
    "/projects/steiflab/scratch/leli/trackastra/A138974A/PrintRun_Apr1223_1311/tracked_postprocessed_2.0/",
    "/projects/steiflab/scratch/leli/trackastra/A146237A/PrintRun_Mar2824_1524/tracked_postprocessed_2.0/",
    "/projects/steiflab/scratch/leli/trackastra/A118880/PrintRun_Jan2624_1252/tracked_again_postprocessed_2.0/", 
    "/projects/steiflab/scratch/leli/trackastra/A138856A/htert_20230822_131349_843.Run/tracked_postprocessed_2.0/"
]
png_directories = [
    "/projects/steiflab/scratch/leli/trackastra/A138974A/PrintRun_Apr1223_1311/tracked_postprocessed_2.0_imgs",
    "/projects/steiflab/scratch/leli/trackastra/A146237A/PrintRun_Mar2824_1524/tracked_postprocessed_2.0_imgs",
    "/projects/steiflab/scratch/leli/trackastra/A118880/PrintRun_Jan2624_1252/tracked_again_postprocessed_2.0_imgs", 
    "/projects/steiflab/scratch/leli/trackastra/A138856A/htert_20230822_131349_843.Run/tracked_postprocessed_2.0_imgs"
]
output_directory = f'/projects/steiflab/scratch/leli/trackastra/feat_extract_analysis''''

# Violin plot for all tracks in a dataset 
# features = feature_extraction_across_tracks(track_info_files[0], tif_directories[0], png_directories[0], output_directory = output_directory, dataset = "A138974A")

# Line plot for all frames in a track
# feature_extraction_across_frames(track_info_file, tif_directory, png_directory, features = ['area',], track = 'not all', track_ids = [6, 217, 253, 133, 175, 142, 191, 258, 289, 271, 341, 380, 425], output_directory = output_directory)

# Violin Plot across selected datasets 
# feature_extraction_across_dataset(track_info_files, tif_directories, png_directories, dataset_names, output_directory = output_directory)