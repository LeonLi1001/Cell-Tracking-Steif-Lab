#!/bin/bash
#SBATCH --job-name=rcnn_pred_1252_cpu      # Job name
#SBATCH --partition=upgrade                # Change this if 'upgrade' isn't suitable for your setup
#SBATCH --ntasks=1                         # Run on a single CPU
#SBATCH --cpus-per-task=2                  # Adjust as necessary
#SBATCH --mem=8000                         # Adjust memory as necessary
#SBATCH --output=/projects/steiflab/scratch/leli/slurm_output/rcnn_pred_1252_cpu_%j.o # Standard log
#SBATCH --error=/projects/steiflab/scratch/leli/slurm_output/rcnn_pred_1252_cpu_%j.e  # Standard error
#SBATCH --workdir=/projects/steiflab/scratch/leli/         # Working directory


chip_name="A138856A"   # Pass chip name as the first command line argument
print_run_name="10dropRun4"   # Pass print run name as the second command line argument

cd /projects/steiflab/scratch/leli/Segmentation
mkdir /projects/steiflab/scratch/leli/slurm_output/$chip_name/$print_run_name
# this is easier to do on a cpu
#loop over every image in the specified directory
#image_directory="/projects/steiflab/archive/data/imaging/${chip_name}/NozzleImages/${print_run_name}"
image_directory="/projects/steiflab/archive/data/imaging/${chip_name}/NozzleImages/${print_run_name}"

for image_filepath in ${image_directory}/*; do
    
    # Submit a job for each image
    sbatch --job-name=rcnn_pred_img --output="/projects/steiflab/scratch/leli/slurm_output/$chip_name/$print_run_name/$(basename $image_filepath).o" --wrap="python PredictRCNNCPU_ll.py $image_filepath $chip_name $print_run_name"

    echo "successfully run one img"

done

