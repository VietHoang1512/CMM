import os
import os.path
import json
import hashlib

import torch
import torchvision.datasets as datasets
from torch.utils.data import ConcatDataset


class _OfficeHome:
    
    DOMAINS = "art  clipart  product  real_world".split()
    CLASSES = "Alarm_Clock  Bottle      Chair       Desk_Lamp  File_Cabinet  Glasses   Knives      Mop       Pan           Printer       Scissors     Soda       ToothBrush Backpack     Bucket      Clipboards  Drill      Flipflops     Hammer    Lamp_Shade  Mouse     Paper_Clip    Push_Pin      Screwdriver  Speaker    Toys Batteries    Calculator  Computer    Eraser     Flowers       Helmet    Laptop      Mug       Pen           Radio         Shelf        Spoon      Trash_Can Bed          Calendar    Couch       Exit_Sign  Folder        Kettle    Marker      Notebook  Pencil        Refrigerator  Sink         Table      TV Bike         Candles     Curtains    Fan        Fork          Keyboard  Monitor     Oven      Postit_Notes  Ruler         Sneakers     Telephone  Webcam".split()
    default_class_order = list(range(len(CLASSES)))
    default_domain_order = list(range(len(DOMAINS)))    
    @classmethod
    def get_complementary_domains(cls, domains):
        return [x for x in cls.DOMAINS if x not in domains]
    
    @classmethod
    def get_complementary_classes(cls, classes):
        return [x for x in cls.CLASSES if x not in classes]    
    
    @classmethod    
    def get_md5(cls, subset_config):
        return hashlib.md5(json.dumps(subset_config).encode('utf-8')).hexdigest()    
    
    def __init__(self, root, train=True, transform=None, target_transform=None, subset_config=None):
        """
        root: path where the 'image-clef' folder is located.
        train: if True, load from 'image-clef/train', else from 'image-clef/test'.
        transform: torchvision transforms to apply.
        subset_config: a dictionary with keys 'domains' and 'classes'. For example:
                       {
                         'domains': ['b', 'c'],
                         'classes': ['aeroplane', 'bike', 'bird', ...]
                       }
        """
        # Base dataset path
        self.root = os.path.join(root, 'OfficeHome')
        
        # Use default configuration if none provided
        if not subset_config:
            subset_config = {'domains': self.DOMAINS, 'classes': self.CLASSES}
        assert 'domains' in subset_config and 'classes' in subset_config, \
            "subset_config must have keys 'domains' and 'classes'"
        print(f"Using subset config: {subset_config}")

        # Use the provided classes as the global class ordering.
        GLOBAL_CLASSES = subset_config['classes']
        global_class_to_idx = {cls_name: i for i, cls_name in enumerate(GLOBAL_CLASSES)}

        # Select the subfolder (train/test)
        subset_dir = 'train' if train else 'test'
        domain_datasets = []

        for domain in subset_config['domains']:
            domain_path = os.path.join(self.root, subset_dir, domain)
            print("Loading data from", domain_path)
            if not os.path.isdir(domain_path):
                print(f"Warning: Domain path '{domain_path}' not found. Skipping.")
                continue

            # Load the domain dataset using ImageFolder.
            ds = datasets.ImageFolder(domain_path, transform=transform)

            # Remap the labels in ds.samples to the global class indices.
            new_samples = []
            for path, local_label in ds.samples:
                local_class_name = ds.classes[local_label]
                if local_class_name in global_class_to_idx:
                    new_samples.append((path, global_class_to_idx[local_class_name]))
                else:
                    print(f"Warning: Class '{local_class_name}' not in global classes. Skipping file {path}.")
            ds.samples = new_samples
            ds.classes = GLOBAL_CLASSES
            ds.class_to_idx = global_class_to_idx
            ds.targets = [s[1] for s in new_samples]
            domain_datasets.append(ds)

        # If only one domain dataset exists, use it directly; else, combine them.
        if len(domain_datasets) == 1:
            self.data = domain_datasets[0]
        else:
            self.data = ConcatDataset(domain_datasets)

    def __len__(self):
        return len(self.data)


class OfficeHome:
    BASE_CLASS = _OfficeHome
    default_class_order = list(range(len(BASE_CLASS.CLASSES)))
    default_domain_order = list(range(len(BASE_CLASS.DOMAINS)))
        
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16,
                 subset_config=None):
        # Create training dataset and DataLoader.
        self.train_dataset = self.BASE_CLASS(
            root=location,
            train=True,
            transform=preprocess,
            target_transform=None,
            subset_config=subset_config,
        ).data
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Create testing dataset and DataLoader.
        self.test_dataset = self.BASE_CLASS(
            root=location,
            train=False,
            transform=preprocess,
            target_transform=None,
            subset_config=subset_config,
        ).data
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        # For ConcatDataset, extract class names from one of the subdatasets.
        if hasattr(self.train_dataset, 'class_to_idx'):
            self.classnames = [c.replace('_', ' ') for c in list(self.train_dataset.class_to_idx.keys())]
        else:
            # Assume at least one sub-dataset has class_to_idx attribute.
            for ds in self.train_dataset.datasets:
                if hasattr(ds, 'class_to_idx'):
                    self.classnames = [c.replace('_', ' ') for c in list(ds.class_to_idx.keys())]
                    break
            else:
                self.classnames = []
