import sys
import os
import numpy as np
import torch
import cv2

sys.path.insert(0, "sam2")

from draw import save_masks, gen_image_writer
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 1. Initialize the model and predictor
print("Initializing SAM2 model and predictor...")
sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=torch.device("cuda"))
predictor = SAM2ImagePredictor(sam2_model)
predictor.speedup()
print("Initialization complete.")

# 2. Prepare batch input data
image1_path = "./sam2/notebooks/images/truck.jpg"
image2_path = "./sam2/notebooks/images/cars.jpg"

if not os.path.exists(image1_path) or not os.path.exists(image2_path):
    print(f"Error: Image files not found.")
    print(f"Please make sure '{image1_path}' and '{image2_path}' exist.")
    sys.exit(1)
    
print(f"Loading images for batch processing...")
image1 = cv2.cvtColor(cv2.imread(image1_path), cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(cv2.imread(image2_path), cv2.COLOR_BGR2RGB)

# Put the images into a list
images_batch = [image1, image2]

# Prepare corresponding point prompts for each image in the batch
points1 = np.array([[500, 375], [502, 375]])
labels1 = np.array([1, 1])
points2 = np.array([[450, 480]])
labels2 = np.array([1])

# Put all points and labels into separate lists to match the `predict_batch` input format
points_batch_list = [points1, points2]
labels_batch_list = [labels1, labels2]

# 3. Use set_image_batch and predict_batch
print("\nSetting image batch for preprocessing...")
predictor.set_image_batch(images_batch)
print(f"Batch of {len(images_batch)} images has been set and preprocessed.")

output_dir = "data/test_image_batch"
os.makedirs(output_dir, exist_ok=True)
print(f"Results will be saved in '{output_dir}' directory.")

print("\n---> Running prediction for the entire batch...")

# Call predict_batch once, passing in the list of prompts
# It will return a list containing the results for all images
all_masks, all_scores, all_logits = predictor.predict_batch(
    point_coords_batch=points_batch_list,
    point_labels_batch=labels_batch_list,
    multimask_output=True,
)
print("Batch prediction complete.")

# 4. Loop through, process, and save the results
print("\nProcessing and saving results...")
# Loop through the returned list of results
for i, (image, masks, scores) in enumerate(zip(images_batch, all_masks, all_scores)):
    # Sort the results for the current image by score
    sorted_ind = np.argsort(scores)[::-1]
    masks_sorted = masks[sorted_ind]
    scores_sorted = scores[sorted_ind]
    
    print(f"  - Image {i}: Found {len(masks_sorted)} masks. Best score: {scores_sorted[0]:.4f}")

    output_filename_base = f"{output_dir}/result_image_{i}"
    save_masks(image, masks_sorted[0:1], deal_func=gen_image_writer(output_filename_base))

print("\nBatch processing test finished successfully.")