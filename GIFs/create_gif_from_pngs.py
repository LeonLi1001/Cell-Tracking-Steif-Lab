# conda activate viewer_env 
import imageio.v2 as imageio
import os
import re 

def numerical_sort(value):
    """
    Extracts the numeric part from the filename for sorting.
    Assumes that the filename format is '<number>_htert_Run'.
    """
    parts = re.findall(r'\d+', value)
    return int(parts[0]) if parts else value

def create_gif_from_pngs(input_directory, output_file, start, end, duration=1):
    """
    Create a GIF from multiple PNG files in a directory.

    :param input_directory: Path to the directory containing PNG files.
    :param output_file: Path to the output GIF file.
    :param duration: Duration (in seconds) for each frame in the GIF.
    """
    # Get a sorted list of PNG files in the directory
    #print(os.listdir(input_directory))
    png_files = sorted([os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith('.png') and not f.startswith('._') and numerical_sort(f) >= start and numerical_sort(f) <= end])
    print(png_files)
    # Read each image and append it to a list
    images = [imageio.imread(png) for png in png_files]

    # Create and save the GIF
    imageio.mimsave(output_file, images, fps=duration, loop = 0)

# Example usage
start = 510
end = 531
dataset = "A138856A"
input_directory = "/projects/steiflab/scratch/leli/trackastra/A138856A/10dropRun3/tracked_again_postprocessed_2.0_imgs"
output_file = f"/projects/steiflab/scratch/leli/{dataset}_{start}_{end}.gif"
create_gif_from_pngs(input_directory, output_file, start, end)
