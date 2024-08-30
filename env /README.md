# Conda Environment for TRACKASTRA

This folder contains the conda environments used: 
 - 'trackastra.yml': was used for the tracking pipeline and all other analysis.. 
 - 'rcnn_env' was used solely for segmentation. 

## The following outlines the usage for conda environment trackastra: 

NOTE: The actual trackastra python package was edited and the edited version is in the folder '/trackastra/'. The following changes should be made if you decide to pull from the original git page. 

for 'model/model.py'

  - old version line(376-377): 
    ```python
    min_time = coords[:, :, :1].min(dim=1, keepdims=True).values
    coords = coords - min_time
    ```
  
  - edited version: 
    ```python
    if coords.shape[1] != 0:
      min_time = coords[:, :, :1].min(dim=1, keepdims=True).values
      coords = coords - min_time
    else: 
      print("coords shape before min_time:  :() ", coords.shape[1])
    ```
for 'tracking/tracking.py'
  - old version (line 137): 
    ```python
      pi = np.stack(pi)
    ```
  - edited version: 
    ```python
    if pi: 
      pi = np.stack(pi)
    else: 
      print("pi is empty here...")
      continue
    ```
  - old version (line 141): 
    ```python
      pj = np.stack(pj)
    ```
  - edited version: 
    ```python
    if pj: 
      pj = np.stack(pj)
    else: 
      print("pj is empty here...")
      continue
    ```

## Environment File

- **trackastra.yml:** 
  - This file defines the Conda environment used for the TRACKASTRA tracking algorithms.
  - It includes all necessary dependencies, packages, and configurations required to run the scripts in this project.

## Testing and Compatibility

- **Tested On:**
  - The environment has been tested on `dlhost12`, ensuring compatibility with the specific hardware and software configurations of this system.

- **CUDA Version:**
  - The environment has been tested with CUDA version **12.0**. Ensure that your system has the appropriate CUDA version installed for optimal performance.

## Setting Up the Environment

To create the Conda environment on your local machine, follow these steps:

1. **Navigate to the `env/` directory:**
   ```bash
   cd env/

2. **Create the environment using trackastra.yml:**
    ```bash
    conda env create -f trackastra.yml
3. **Activate the environment:**
    ```bash
    conda activate trackastra