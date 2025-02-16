import os

def required_files(num_splits: int):
    """
    Return a set of required filenames for a given number of splits.
    For example, if num_splits=10, then the set will contain:
    {
      'finetuned_0.pt', 'finetuned_1.pt', ..., 'finetuned_9.pt',
      'fisher_0.pt',    'fisher_1.pt',    ..., 'fisher_9.pt'
    }
    """
    files = set()
    for i in range(num_splits):
        files.add(f"finetuned_{i}.pt")
        files.add(f"fisher_{i}.pt")
    return files

def main():
    # Update this path to wherever you want to start the search
    root_dir = "."

    # Define how many splits to look for
    splits_list = [10, 20, 50]

    # Precompute the required files for each splits value
    split_requirements = {
        splits: required_files(splits) for splits in splits_list
    }

    # Walk through the directory tree
    for root, dirs, files in os.walk(root_dir):
        file_set = set(files)

        # Check each n_splits possibility
        for splits in splits_list:
            if f"n_splits={splits}" in root:
                # Check if the current directory contains all required files
                req_files = split_requirements[splits]
                if req_files.issubset(file_set):
                    print(f"Found all required files for n_splits={splits} in: {root}")
                    # Delete each of the required files
                    for f in req_files:
                        file_path = os.path.join(root, f)
                        try:
                            os.remove(file_path)
                            print(f"Deleted {file_path}")
                        except OSError as e:
                            print(f"Error deleting {file_path}: {e}")

if __name__ == "__main__":
    main()
