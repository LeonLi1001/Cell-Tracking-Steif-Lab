from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import sys
import os
import numpy as np
import cv2
from PIL import Image

# Constants and dictionary
SCRATCH = "/projects/steiflab/scratch/leli"
ARCHIVE = "/projects/steiflab/archive/data/imaging"
CHIPS = {
    'A138974A': ['PrintRun_Apr1223_1311'],

    'A138856A': ['10dropRun1', '10dropRun2', '10dropRun3', '10dropRun4',
                 '5dropRun1', '5dropRun2', '5dropRun3', '5dropRun4', '5dropRun5', 'htert_20230822_131349_843.Run'],

    'testing_100':['testing_run_100'],
    'A146270A':['PrintRun_Mar2124_1513'],
    'A146237A': ['PrintRun_Mar2824_1524', 'PrintRun_Mar2824_1528', 'PrintRun_Mar2824_1539', 'PrintRun_Mar2824_1547', 'PrintRun_Mar2824_1550', 'PrintRun_Mar2824_1704', 'PrintRun_Mar2824_1705', 
                    'PrintRun_Mar2824_1707', 'PrintRun_Mar2824_1708', 'PrintRun_Mar2824_1709', 'PrintRun_Mar2824_1714', 'PrintRun_Mar2824_1717'],

    'A118880': ['PrintRun_Jan2624_1252'],
}

if len(sys.argv) != 4:
    print("Usage: python PredictRCNN.py imgPath chipName runName") 
    sys.exit()

img_path, chip_name, run_name = sys.argv[1:4]

if chip_name not in CHIPS or run_name not in CHIPS[chip_name]:
    print(f"Error: Invalid chip or run. Provided chip '{chip_name}' or run '{run_name}' not found in the dictionary.")
    sys.exit(1)

if "Printed" in img_path or not img_path.endswith("_htert_Run.png"): 
    print(f"Error: This is a duplicate with the filename: {os.path.basename(img_path)}")
    sys.exit(1)

#model_weights_path = os.path.join(SCRATCH, chip_name, run_name, "model_final.pth")
model_weights_path = os.path.join("/projects/steiflab/scratch/wgervasio/A138974A/PrintRun_Apr1223_1311/", "model_final.pth")
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_weights_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.DEVICE = 'cpu'

predictor = DefaultPredictor(cfg)

# Load the image
img = cv2.imread(img_path)
img = np.rot90(img)
outputs = predictor(img)

# Visualization
MetadataCatalog.get("custom_dataset_train").set(thing_classes=["cell"])
visualizer = Visualizer(img[:, :, ::-1], MetadataCatalog.get('custom_dataset_train'), scale=1.2)
out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))

# Save overlayed image
output_directory_overlayed = os.path.join(SCRATCH, chip_name, run_name, "rcnn_output_overlayed")
os.makedirs(output_directory_overlayed, exist_ok=True)
output_file_path_overlayed = os.path.join(output_directory_overlayed, os.path.basename(img_path))
cv2.imwrite(output_file_path_overlayed, out.get_image())

# Save individual masks
output_directory_masks = os.path.join(SCRATCH, chip_name, run_name, "rcnn_output_masks", os.path.splitext(os.path.basename(img_path))[0])
os.makedirs(output_directory_masks, exist_ok=True)
masks = outputs["instances"].pred_masks.to("cpu").numpy()

for idx, mask in enumerate(masks):
    mask_image = (mask * 255).astype(np.uint8)
    mask_filename = f"{idx}.png"
    mask_filepath = os.path.join(output_directory_masks, mask_filename)
    cv2.imwrite(mask_filepath, mask_image)
