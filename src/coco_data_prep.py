import os
import random
import shutil
from pycocotools.coco import COCO

# Paths
coco_images_dir = './data/train2017'  # Path to COCO images folder
coco_annotations_path = './data/annotations/instances_train2017.json'  # Path to COCO annotations JSON
output_dir = './data/subset'  # Path to store the selected subset

# Initialize COCO API
coco = COCO(coco_annotations_path)

# Get all image IDs (no category filtering)
image_ids = coco.getImgIds()

# Print total available images
print(f"Total available images: {len(image_ids)}")

# Number of images to select
num_images = 10000  # You can change this as needed

# Ensure num_images does not exceed available images
num_images = min(num_images, len(image_ids))

# Randomly sample images
selected_image_ids = random.sample(image_ids, num_images)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Copy images to the output directory
for img_id in selected_image_ids:
    img_info = coco.loadImgs(img_id)[0]
    img_file = img_info['file_name']
    src_path = os.path.join(coco_images_dir, img_file)
    
    if os.path.exists(src_path):
        dest_path = os.path.join(output_dir, img_file)
        shutil.copy(src_path, dest_path)
        print(f"Copied {img_file} to {output_dir}")
    else:
        print(f"Image {img_file} not found in {coco_images_dir}")
