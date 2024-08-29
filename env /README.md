# Conda Environment for TRACKASTRA

This folder contains the Conda environment file used for running the TRACKASTRA-based tracking algorithms and related scripts in this repository.

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