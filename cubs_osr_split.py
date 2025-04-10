import random
from pathlib import Path
from collections import defaultdict

# Config
seed = 42
num_id_classes = 150
val_frac = 0.1
test_frac = 0.2

# Paths
cub_dir = Path("data/CUB_200_2011")
images_file = cub_dir / "images.txt"
labels_file = cub_dir / "image_class_labels.txt"
classes_file = cub_dir / "classes.txt"

# Load class mappings
class_id_to_name = {}
with classes_file.open() as f:
    for line in f:
        class_id, class_name = line.strip().split()
        class_id_to_name[int(class_id)] = class_name

# Load image paths
image_id_to_path = {}
with images_file.open() as f:
    for line in f:
        image_id, path = line.strip().split()
        image_id_to_path[int(image_id)] = path

# Load image labels
image_id_to_label = {}
with labels_file.open() as f:
    for line in f:
        image_id, class_id = line.strip().split()
        image_id_to_label[int(image_id)] = int(class_id)

# Build class_id -> image paths mapping
class_to_images = defaultdict(list)
for img_id, class_id in image_id_to_label.items():
    img_path = image_id_to_path[img_id]
    class_to_images[class_id].append(img_path)

# Shuffle and split class IDs
all_class_ids = sorted(class_to_images.keys())
random.seed(seed)
random.shuffle(all_class_ids)

id_class_ids = sorted(all_class_ids[:num_id_classes])
ood_class_ids = sorted(all_class_ids[num_id_classes:])

# Map original class IDs to 0–(num_id_classes-1)
id_class_id_map = {orig_id: i for i, orig_id in enumerate(id_class_ids)}

# Containers
train_data, val_data, test_id_data, test_ood_data = [], [], [], []

# Split ID classes
for orig_class_id in id_class_ids:
    class_idx = id_class_id_map[orig_class_id]
    images = class_to_images[orig_class_id]
    random.shuffle(images)
    n = len(images)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)

    val_data += [(img, class_idx) for img in images[:n_val]]
    test_id_data += [(img, class_idx) for img in images[n_val:n_val + n_test]]
    train_data += [(img, class_idx) for img in images[n_val + n_test:]]

# OOD samples (label = -1)
for orig_class_id in ood_class_ids:
    test_ood_data += [(img, -1) for img in class_to_images[orig_class_id]]

# Save function
# Set base directory correctly as a Path object
output_dir = Path('data/benchmark_imglist/osr_cub150')
dirs = ['train', 'val', 'test']

# Create subdirectories
for d in dirs:
    dir_path = output_dir / d
    dir_path.mkdir(parents=True, exist_ok=True)

# Save function with Path handling
def save_split(filename, data):
    filepath = output_dir / filename  # now safe to use /
    with filepath.open("w") as f:
        for img_path, label in data:
            full_path = f"{cub_dir}/images/{img_path}"
            f.write(f"{full_path} {label}\n")

# Save all splits to their respective folders
save_split("train/train_cub200_150_seed1.txt", train_data)
save_split("val/val_cub200_150_seed1.txt", val_data)
save_split("test/test_cub200_150_id_seed1.txt", test_id_data)
save_split("test/test_cub200_50_ood_seed1.txt", test_ood_data)

print("✅ Finished saving all splits to", output_dir)