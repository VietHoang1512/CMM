import os
import random
import shutil
from tqdm.auto import tqdm
# --- CONFIGURABLE PARAMETERS ---
SOURCE_FOLDER = "/vast//data/Office31"
DOMAINS = "amazon  dslr webcam".split()  # The subfolders in your dataset
TRAIN_FOLDER = SOURCE_FOLDER + "/train"
TEST_FOLDER = SOURCE_FOLDER + "/test"
SPLIT_RATIO = 0.2  # 20% for test, 80% for train

# Optional: set a random seed if you want reproducible splits
random.seed(42)

def ensure_dir_exists(path):
    """Create the directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

for domain in DOMAINS:
    domain_path = os.path.join(SOURCE_FOLDER, domain)
    if not os.path.isdir(domain_path):
        continue  # Skip if the domain folder doesn't exist

    # Loop over each class folder in the domain
    for class_name in os.listdir(domain_path):
        class_path = os.path.join(domain_path, class_name)
        if not os.path.isdir(class_path):
            continue  # Skip if it's not a folder

        # Collect all image filenames
        images = [f for f in os.listdir(class_path) 
                  if os.path.isfile(os.path.join(class_path, f))]
        
        # Shuffle images to randomize
        random.shuffle(images)

        # Compute how many go into test vs train
        test_size = int(len(images) * SPLIT_RATIO)
        test_images = images[:test_size]
        train_images = images[test_size:]

        # Create corresponding train/test class folders
        train_class_path = os.path.join(TRAIN_FOLDER, domain, class_name)
        test_class_path = os.path.join(TEST_FOLDER, domain, class_name)
        ensure_dir_exists(train_class_path)
        ensure_dir_exists(test_class_path)

        for img in tqdm(train_images, total=len(train_images), desc="Copy train examples"):
            src = os.path.join(class_path, img)
            dst = os.path.join(train_class_path, img)
            shutil.copy2(src, dst)

        # Copy test images
        for img in tqdm(test_images, total=len(test_images), desc="Copy test examples"):
            src = os.path.join(class_path, img)
            dst = os.path.join(test_class_path, img)
            shutil.copy2(src, dst)

print("Dataset split completed!")



