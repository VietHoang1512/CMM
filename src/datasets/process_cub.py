import os
import shutil

# Set this to the path where your original CUB_200_2011 folder is located.
location = '/vast/data/CUB_200_2011'  # <-- change this!


# Define paths to relevant files and directories.
images_dir = os.path.join(location, 'images')
images_txt_path = os.path.join(location, 'images.txt')
split_txt_path = os.path.join(location, 'train_test_split.txt')

# Destination base directories for the split images.
base_dest = os.path.join(location, 'CUB_200_2011_splitted', 'images_train_test')
traindir = os.path.join(base_dest, 'train')
valdir = os.path.join(base_dest, 'val')

# Create the base destination directories if they don't exist.
os.makedirs(traindir, exist_ok=True)
os.makedirs(valdir, exist_ok=True)

# Build a mapping from image id to its relative path.
id_to_path = {}
with open(images_txt_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            img_id, rel_path = parts
            id_to_path[img_id] = rel_path

# Process the train_test_split file to copy images to the correct folder.
with open(split_txt_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 2:
            continue  # skip malformed lines
        img_id, is_train = parts

        if img_id not in id_to_path:
            print(f"Warning: image id {img_id} not found in images.txt")
            continue

        # Get the relative image path and compute the full source path.
        rel_img_path = id_to_path[img_id]
        src_img_path = os.path.join(images_dir, rel_img_path)

        # Extract the class name from the relative path.
        # For example, if rel_img_path is "200.Common_Yellowthroat/Common_Yellowthroat_0114_190501.jpg",
        # then the class folder is "200.Common_Yellowthroat".
        class_name = os.path.dirname(rel_img_path)

        # Decide whether the image goes into the training or validation folder.
        if is_train == '1':
            dst_base = traindir
        else:
            dst_base = valdir

        # Create the class subfolder inside the destination if it doesn't exist.
        class_dir = os.path.join(dst_base, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Set the destination image path.
        dst_img_path = os.path.join(class_dir, os.path.basename(src_img_path))
        
        # Optionally, print what is happening.
        print(f"Copying {src_img_path} to {dst_img_path}")

        # Copy the image file.
        shutil.copy(src_img_path, dst_img_path)

print("Finished copying images to train and val directories with class subfolders.")