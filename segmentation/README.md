# Segmentation

This directory contains the necessary scripts and batch files to perform segmentation on your dataset. The segmentation process is the first step in the pipeline.

## Running the Segmentation Script

### Step 1: Activate the Conda Environment

Before running the segmentation, make sure to activate the `rcnn_env` environment. This environment includes all the necessary dependencies for running the segmentation scripts.

```bash
conda activate rcnn_env
```

### Step 2: Navigate to the Segmentation Directory
Change your current working directory to the segmentation/ folder:

```bash
cd segmentation
```

### Step 3: Set Up the Batch Script and Python Script
Before running the batch script, you need to ensure that both the batch file and the associated Python script have the correct dataset and printrun values.

 - Edit the Batch Script (run_rcnn_preds_cpu.sh):
  - Open the batch file in your text editor.
  - Modify the dataset and printrun values according to your specific experiment.
 - Edit the Python Script:

Similarly, ensure that the dataset and printrun values are correctly set at the beginning of the associated Python script.

### Step 4: Run the Batch Script
Once the correct values have been inputted, you can run the batch script using the following command:

```bash
sbatch run_rcnn_preds_cpu.sh
```

##CellenONE Specific Scripts
The scripts and batch files with CellenONE in their names are specifically designed to process outputs that were produced by the CellenONE technology. Make sure to use these scripts if you are working with CellenONE datasets.