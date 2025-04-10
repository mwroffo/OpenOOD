import os

IMG_COUNT = 11788
train_count = 0
cub_image_classic_dir = './data/images_classic/cub200'
train_dir = cub_image_classic_dir+'/train/'
class_dirs = os.listdir(train_dir)
for class_dir in class_dirs:
    files = os.listdir(train_dir + class_dir)
    train_count += len(files)
    
print("Train count", train_count)

test_count = 0
test_dir = cub_image_classic_dir+'/test/'
class_dirs = os.listdir(test_dir)
for class_dir in class_dirs:
    files = os.listdir(test_dir + class_dir)
    test_count += len(files)
    
print("Test count", test_count)
print("Total", test_count + train_count)
if test_count + train_count != IMG_COUNT:
    print("WRONG NUMBER OF IMAGES IN EACH DIRECTORY")