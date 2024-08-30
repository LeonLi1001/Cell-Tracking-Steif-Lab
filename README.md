This repository contains various components and scripts used in different stages of our project, from tracking algorithms to feature extraction and image generation.

## Repository Structure

- **env/**
  - This folder contains the Conda environment used in this project. It includes all the dependencies and packages required to run the scripts and algorithms within this repository.

- **segmentation/**
  - This folder contains scripts used for the segmentation process, which is the first step in the pipeline. It includes the necessary batch script for running segmentation on the dataset.

- **Tracker/**
  - This folder includes the tracking algorithm and the post-processing scripts that are ready for use. 

- **Track_to_well/**
  - The scripts in this folder are used to link the tracks generated by our tracker to the specific wells used in the experiment. 

- **Feature_extraction/**
  - This folder contains scripts for downstream analysis, specifically for extracting morphological features from the tracks generated by the Tracker. 

- **Isolatrix_figure/**
  - The Isolatrix folder holds all the scripts used to generate images for the Isolatrix paper. 

## Getting Started
1. **Set Up the Segmentation Environment**

Before proceeding with the steps below, you need to set up the segmentation environment and run the segmentation process.

  - Set Up the Conda Environment:**
  - Navigate to the `env/` directory and create the `rcnn_env` environment using the provided environment file:
    ```bash
    conda env create -f rcnn_env.yml
    ```
  - Activate the environment:
    ```bash
    conda activate rcnn_env
    ```

2. **Run the Segmentation Script**

  - Run the Batch Script:**
  - Navigate to the `segmentation/` folder.
  - Execute the batch script `run_rcnn_preds_cpu.sh` to perform the segmentation:
    ```bash
    ./run_rcnn_preds_cpu.sh
    ```

1. **Set up the Tracking Environment:**
   - Use the environment files in the `env/` folder to recreate the Conda environment required to run the scripts. You can do this by navigating to the `env/` directory and using the command:
     ```bash
     conda env create -f trackastra.yml
     conda activate <trackastra>
     ```

2. **Run Tracking Algorithms:**
   - Navigate to the `Tracker/` folder to run the tracking algorithms with post-processing. 

3. **Link Tracks to Wells:**
   - Use the scripts in `Track_to_well/` to associate the generated tracks with the corresponding experimental wells. 

4. **Feature Extraction:**
   - For downstream analysis, run the scripts in `Feature_extraction/` to extract morphological features from the tracked data. These features will be used in further analysis to derive insights from the experiments.

5. **Generate Isolatrix Figures:**
   - The `Isolatrix_figure/` folder contains a script that was used to generate the figures for the Isolatrix paper. Run the code block to produce the required visualizations.

## Acknowledgments
- This repository was developed in the Steif Lab under the supervision of Adi Steif. For more information, visit the [Steif Lab website](https://mavis.bcgsc.ca/labs/steif-lab).

- **TRACKASTRA:** This project uses the `TRACKASTRA` package for the tracking algorithms. We would like to acknowledge the developers of `TRACKASTRA` for their work. For more information about `TRACKASTRA`, visit the [TRACKASTRA GitHub repository](https://github.com/weigertlab/trackastra).

- **Segmentatino** part of the repo is done by Will Gervasio, a former lab member.

## Contributing

If you wish to contribute to this project, please follow the standard Git workflow: fork the repository, make your changes, and submit a pull request.


