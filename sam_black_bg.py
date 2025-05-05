import numpy as np
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path
import os

'''
This script generates a synthetic version of the CUB-200-2011 dataset by applying background masking 
using the Segment Anything Model (SAM). It creates a copy of each image where the background has 
been blacked out, preserving only the segmented foreground object (typically the bird).

- Loads the SAM ViT-B model checkpoint for image segmentation.
- For each image, uses a center-click prompt to predict the main object mask.
- Applies the mask to remove the background and keeps only the foreground pixels.
- Saves the resulting masked images in a parallel directory:
  data/CUB_200_2011_masked/images/, maintaining the original class structure.
'''

def main():
    # Load SAM
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    sam.to("cuda")
    predictor = SamPredictor(sam)
    
    # List all files and directories in the current working directory
    base_dir = "data/CUB_200_2011/images/"
    masked_base_dir = "data/CUB_200_2011_masked/images/"
    class_dirs = os.listdir("data/CUB_200_2011/images/")
    
    for class_dir in class_dirs:
        imgs = os.listdir(base_dir+class_dir)
        for img in imgs:
            
            masked_image_dir =  masked_base_dir + class_dir + "/"
            masked_img_path = masked_image_dir + img
            
            if not os.path.exists(masked_img_path):
                img_path = base_dir+class_dir+ "/" + img
                image = np.array(Image.open(img_path).convert("RGB"))
                masked = mask_image(predictor, image)
            
                output_dir = Path(masked_image_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(masked_img_path, masked)
            
    
def mask_image(predictor, image):
    predictor.set_image(image)
    input_point = np.array([[image.shape[1] // 2, image.shape[0] // 2]])  # center point
    input_label = np.array([1])
    masks, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)

    # Apply mask
    masked = image.copy()
    masked[masks[0] == 0] = 0
    return masked
    
if __name__ == '__main__':
    main()