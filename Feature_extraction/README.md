# Feature Extraction for Tracking Data

This directory contains scripts designed to extract various morphological and intensity features from tracking data. The extracted features can be used for downstream analysis, including generating statistical plots like histograms, boxplots, and violin plots.

## Scripts Overview

### Feature Extraction Functions

1. **get_diameter_cellenONE(frame, track_id, um_per_pixel=0.69):**
   - Calculates the diameter of an object in the frame associated with a given `track_id`.
   - Returns the diameter in micrometers.

2. **get_elongation_cellenONE(frame, track_id):**
   - Computes the elongation of an object, defined as the ratio of the major axis to the minor axis.
   - Returns the elongation ratio.

3. **get_gray_intensity_cellenONE(frame, intensity_image, track_id):**
   - Calculates the mean normalized intensity of an object in the frame.
   - Returns the mean intensity.

4. **get_glcm_features(frame, intensity_image, track_id, get_e):**
   - Calculates Gray Level Co-occurrence Matrix (GLCM) features such as energy and homogeneity for a given `track_id`.
   - Returns the specified GLCM feature.

5. **get_w(frame, track_id) & get_h(frame, track_id):**
   - Calculates the width (`get_w`) and height (`get_h`) of an object in the frame.
   - Returns the width or height in pixels.

6. **get_circ(frame, track_id):**
   - Calculates the circularity of an object, using the `DIPlib` library.
   - Returns the circularity value.

7. **get_intensity(frame, intensity_image, track_id):**
   - Calculates the mean normalized intensity using Z-score normalization.
   - Returns the mean intensity.

### Plotting Functions

1. **linear_plot_with_summary(values, names, title, output_directory=None, xlab='Track ID'):**
   - Plots a linear graph of values with summary statistics in the legend and highlights points outside specified bounds.
   - Saves the plot to the specified output directory.

2. **histogram_with_best_fit(values, title, xlab, output_directory=None, dataset=None):**
   - Plots a histogram with a best-fit line and summary statistics.
   - Saves the plot to the specified output directory.

3. **create_boxplot(data, xticklabels, title, output_directory=None):**
   - Creates a boxplot for multiple datasets and labels the x-axis with `xticklabels`.
   - Saves the plot to the specified output directory.

4. **create_violinplot(data, xticklabels, title, output_directory=None):**
   - Generates a violin plot for multiple datasets.
   - Saves the plot to the specified output directory.

5. **create_violinplot_with_boxplot(data, xticklabels, title, output_directory=None):**
   - Generates a violin plot with embedded boxplots for multiple datasets.
   - Saves the plot to the specified output directory.

### Feature Extraction Across Tracks and Frames

1. **feature_extraction_across_tracks():**
   - Extracts features for all or specified tracks across frames in a dataset.
   - Outputs a DataFrame containing features such as area, length, width, circularity, intensity, energy, homogeneity, diameter, elongation, and intensity_not_norm.

2. **feature_extraction_across_frames():**
   - Extracts features for each frame in specified tracks and plots them over time.
   - Generates line plots showing the change in features across frames.

3. **feature_extraction_across_dataset():**
   - Extracts features across multiple datasets and compares them using violin plots.
   - Supports extracting features such as area, length, width, circularity, intensity, energy, homogeneity, diameter, elongation, and intensity_not_norm across multiple datasets.

## Example Usage
shown in detail at the end of the script. 