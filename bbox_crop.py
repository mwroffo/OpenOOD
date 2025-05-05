from PIL import Image

'''
- Reads `images.txt` and `bounding_boxes.txt` to map each image to its bounding box.
- Crops each image using its bounding box with optional padding.
- Saves cropped images in a parallel directory (`data/CUB_200_2011_crop/images/`), maintaining 
  the same class-based folder structure.
- Skips any images without bounding box annotations.
'''
import os

def main():
    # List all files and directories in the current working directory
    base_dir = "data/CUB_200_2011/images/"
    cropped_base_dir = "data/CUB_200_2011_crop/images/"
    
    images_txt = "data/CUB_200_2011/images.txt"
    bbox_txt = "data/CUB_200_2011/bounding_boxes.txt"
    
    image_id_to_path = load_image_id_mapping(images_txt)
    path_to_image_id = {v: k for k, v in image_id_to_path.items()}
    bbox_dict = load_bbox_mapping(bbox_txt)

    
    class_dirs = os.listdir("data/CUB_200_2011/images/")
    
    
    for i, class_dir in enumerate(class_dirs):
        imgs = os.listdir(base_dir+class_dir)
        for img in imgs:
            relative_path = os.path.join(class_dir, img)
            cropped_image_dir = os.path.join(cropped_base_dir, class_dir)
            cropped_img_path = os.path.join(cropped_image_dir, img)
            img_path = os.path.join(base_dir, class_dir, img)
            
            if not os.path.exists(cropped_img_path):
                if relative_path not in path_to_image_id:
                    print(f"⚠️ Skipping {relative_path}, not found in mapping.")
                    continue
                
                image_id = path_to_image_id[relative_path]
                bbox = bbox_dict.get(image_id)
                
                if bbox is None:
                    print(f"⚠️ No bbox for image ID {image_id}")
                    continue
                
                img = Image.open(img_path).convert("RGB")
                cropped = crop_with_bbox(img, bbox)

                os.makedirs(cropped_image_dir, exist_ok=True)
                cropped.save(cropped_img_path)

def load_image_id_mapping(images_txt_path):
    image_id_to_path = {}
    with open(images_txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            image_id = int(parts[0])
            image_path = parts[1]
            image_id_to_path[image_id] = image_path
    return image_id_to_path

def load_bbox_mapping(bbox_txt_path):
    bbox_dict = {}
    with open(bbox_txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            image_id = int(parts[0])
            bbox = list(map(float, parts[1:]))  # x, y, width, height
            bbox_dict[image_id] = bbox
    return bbox_dict

def crop_with_bbox(image: Image.Image, bbox: list, padding=0.1) -> Image.Image:
    x, y, width, height = bbox
    x1 = max(int(x - padding * width), 0)
    y1 = max(int(y - padding * height), 0)
    x2 = min(int(x + width + padding * width), image.width)
    y2 = min(int(y + height + padding * height), image.height)
    return image.crop((x1, y1, x2, y2))

    
if __name__ == '__main__':
    main()