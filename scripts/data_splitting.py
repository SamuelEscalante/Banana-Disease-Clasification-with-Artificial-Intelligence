import os
import numpy as np
from glob import glob
from tqdm import tqdm
import logging as log

# Set up logging

log.basicConfig(level=log.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

# Define class names and number of images per class
class_names = ['cordana', 'healthy', 'pestalotiopsis', 'sigatoka']
n_images_per_class = len(os.listdir(f"AugmentedSet/{class_names[0]}"))

log.info(n_images_per_class)

# Define train, valid, and test directories and create them if they don't exist
train_dir = "data/train"
valid_dir = "data/valid"
test_dir  = "data/test"

for directory in [train_dir, valid_dir, test_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Create subdirectories for each class in train, valid, and test directories
for name in class_names:
    for directory in [train_dir, valid_dir, test_dir]:
        class_path = os.path.join(directory, name)
        if not os.path.exists(class_path):
            os.makedirs(class_path)

# Collect all image paths for each class
all_class_paths = [glob(f"AugmentedSet/{name}/*") for name in class_names]

# Define training, validation, and testing size
total_size = sum([len(paths) for paths in all_class_paths])

train_ratio = 0.70
test_ratio  = 0.20
valid_ratio = 0.10

train_size = int(total_size * train_ratio)
valid_size = int(total_size * valid_ratio)
test_size  = int(total_size * test_ratio)

train_images_per_class = int(n_images_per_class * train_ratio)
valid_images_per_class = int(n_images_per_class * valid_ratio)
test_images_per_class  = int(n_images_per_class * test_ratio)

log.info("Total Data Size  :   {}".format(total_size))
log.info("Training Size    :   {}".format(train_size))
log.info("Validation Size  :   {}".format(valid_size))
log.info("Testing Size     :   {}\n".format(test_size))

# Shuffle image paths for each class
for paths in all_class_paths:
    np.random.shuffle(paths)

# Define lists of (old_path, new_path) tuples for training, validation, and testing images
train_images = [(path, os.path.join(train_dir, os.path.basename(os.path.dirname(path)), os.path.basename(path))) for paths in all_class_paths for path in paths[:train_images_per_class]]
valid_images = [(path, os.path.join(valid_dir, os.path.basename(os.path.dirname(path)), os.path.basename(path))) for paths in all_class_paths for path in paths[train_images_per_class: train_images_per_class + valid_images_per_class]]
test_images  = [(path, os.path.join(test_dir, os.path.basename(os.path.dirname(path)), os.path.basename(path)))  for paths in all_class_paths for path in paths[train_images_per_class+valid_images_per_class: train_images_per_class + valid_images_per_class + test_images_per_class]]

# Move images to their new directories
for images, data_type in [(train_images, "Training"), (valid_images, "Validation"), (test_images, "Testing")]:
    for (old_path, new_path) in tqdm(images, desc=data_type + " Data"):
        os.rename(old_path, new_path)

# Remove the old directories
# for directory in class_names:
#     os.rmdir("OriginalSet/" + directory)

# log.info confirmation message

log.info("ALL DONE!!")