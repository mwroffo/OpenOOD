import numpy as np
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path
import os

'''
This script preprocesses the CUB-200-2011 dataset by selectively blurring image backgrounds 
while keeping the foreground bird sharp. It uses the Segment Anything Model (SAM) to segment 
each image and applies Gaussian blur only to background regions, simulating a depth-of-field effect.

Purpose:
--------
- Enhance fine-grained classification or OSR tasks by promoting subject prominence.
- Suppress potentially misleading background cues to improve generalization.

Key Features:
-------------
- Loads the SAM ViT-B model checkpoint.
- For each image, generates a binary mask from a center-click prompt.
- Applies a blur to the background while preserving the foreground bird.
- Saves the modified images in `data/CUB_200_2011_blur/images/` with class-wise folders.
'''

def main():
    # Load SAM
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    sam.to("cuda")
    predictor = SamPredictor(sam)
    
    # List all files and directories in the current working directory
    base_dir = "data/CUB_200_2011/images/"
    blurred_base_dir = "data/CUB_200_2011_blur/images/"
    class_dirs = os.listdir("data/CUB_200_2011/images/")
    
    for class_dir in class_dirs:
        imgs = os.listdir(base_dir+class_dir)
        for img in imgs:
            blurred_image_dir =  blurred_base_dir + class_dir + "/"
            blurred_img_path = blurred_image_dir + img
            
            if not os.path.exists(blurred_img_path):
                img_path = base_dir+class_dir+ "/" + img
                image = Image.open(img_path)
                blurred_image = blur(predictor, image)
            
                output_dir = Path(blurred_image_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                blurred_image.save(blurred_img_path)

    
def blur(predictor, image):
    image_np = np.array(image.convert("RGB"))
    h, w, _ = image_np.shape

    # Run SAM to get mask
    predictor.set_image(image_np)
    input_point = np.array([[w // 2, h // 2]])  # center click
    input_label = np.array([1])
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

    mask = masks[0].astype(np.uint8)

    # Blur entire image
    blurred = cv2.GaussianBlur(image_np, (21, 21), 0)

    # Composite: keep original foreground, use blurred background
    result = np.where(mask[:, :, None], image_np, blurred)

    return Image.fromarray(result)
    
if __name__ == '__main__':
    main()