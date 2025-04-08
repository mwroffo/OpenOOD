import os
import shutil

# This file will inspect the downloaded CUBs dataset and 
# move the images based on the corresponding train/test split
# into the requisite files required by the OOD framework.
# Make sure the local paths here match where you extracted the 
# downloaded CUBs dataset.

# wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1
# tar -xvzf CUB_200_2011.tgz


# Basepath where you stored extracted CUBs dataset
path= './data/CUB_200_2011/'
# Images folder of CUBs dataset
image_dir = "./data/CUB_200_2011/images/"
# Split file of CUBs dataset
split_file = 'train_test_split.txt'
# File mapping IDs to images
id_file = 'images.txt'


samples = {}

# Read in split file, setting 'is_train' in our samples map
with open(path+split_file, "r") as file:
    for line in file:
        image_id, is_training = line.strip().split()
        if image_id not in samples:
            samples[image_id] = {}
        samples[image_id]['is_train'] = is_training
        
print("Len samples", len(samples))

# Read in id file, setting the 'img_path' in our samples map
with open(path+id_file, "r") as file:
    for line in file:
        image_id, img_path = line.strip().split()
        samples[image_id]['img_path'] = img_path
        
# Need to join the classes.txt in our samples map to get class identifier
# Read in split file, setting 'is_train' in our samples map
classes_file = 'classes.txt'

class_map = {}
with open(path+classes_file, "r") as file:
    for line in file:
        class_id, class_str = line.strip().split()
        class_map[class_id] = class_str

for id, val in samples.items():
    for class_id, class_str in class_map.items():
        if class_str in val['img_path']:
            val['class_id'] = class_id

print('Len of classes', len(class_map))
print("Len samples", len(samples))
print(samples['1'], samples['11788'])

# Make train/test folders for all classes
cub_image_classic_dir = './data/images_classic/cub200'
class_dirs = os.listdir(image_dir)
for class_dir in class_dirs:
    os.makedirs(cub_image_classic_dir+'/train/'+class_dir, exist_ok=True)
    os.makedirs(cub_image_classic_dir+'/test/'+class_dir, exist_ok=True)

# For all our samples, copy the source image to the destination train/test folder of OOD
train_path = cub_image_classic_dir + '/train/'
test_path = cub_image_classic_dir + '/test/'
for id, val in samples.items():
    og_path = image_dir + val['img_path']
    if val['is_train'] == '1':
        # write image to train
        if not os.path.exists(train_path + val['img_path']):
            shutil.copy(og_path, train_path + val['img_path'])
    else:
        # write image to test
        if not os.path.exists(test_path + val['img_path']):
            shutil.copy(og_path, test_path + val['img_path'])
        
# Need to setup the ./data/benchmark_imglist
# Needs a test_cub200.txt and train_cub200.txt
# Each file is a list of dataset/type/class/img class_number
# Ex: cifar10/test/airplane/0298.png 0
output_dir = './data/benchmark_imglist/cub200'
benchmark_train_list = os.path.join(output_dir, 'train_cub200.txt')
benchmark_test_list = os.path.join(output_dir, 'test_cub200.txt')
# Make sure the output dir exists
os.makedirs(output_dir, exist_ok=True)
# Touch the files (optional but makes intent clear)
open(benchmark_train_list, 'a').close()
open(benchmark_test_list, 'a').close()

# Open both files for writing
with open(benchmark_train_list, 'w') as train_file, open(benchmark_test_list, 'w') as test_file:
    for id, val in samples.items():
        if val['is_train'] == '1':
            line = f"cub200/train/{val['img_path']} {val['class_id']}\n"
            train_file.write(line)
        else:
            line = f"cub200/test/{val['img_path']} {val['class_id']}\n"
            test_file.write(line)