import torch
import torch.nn as nn
import numpy as np
import os
import requests
import tarfile
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image


COMMON_CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 
                    'snow', 'frost', 'fog', 'brightness', 'contrast', 
                    'elastic_transform', 'pixelate', 'jpeg_compression']
ALL_CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'speckle_noise', 'impulse_noise',
                    'defocus_blur', 'gaussian_blur', 'motion_blur', 'zoom_blur', 'glass_blur', 
                    'snow', 'fog', 'brightness', 'contrast', 'frost', 
                    'elastic_transform', 'pixelate', 'jpeg_compression', 'spatter', 'saturate']

class CIFAR10C(Dataset):
    """
    Costum dataset class for the CIFAR-10-C dataset.
    """
    def __init__(self, root, corruptions = ['gaussian_noise',], severity : int = 1, 
                 transform = None, target_transform = None, download = True):
        assert 1 <= severity <= 5, "invalid severity level!"
        assert all(c in ALL_CORRUPTIONS for c in corruptions), "invalid corruption type!"

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        n_total_cifar = 10000

        cifar10c_dir = os.path.join(self.root, 'CIFAR-10-C')
        if not os.path.exists(cifar10c_dir):
            print(f"Dataset folder not found: '{cifar10c_dir}'.")
            if download:
                print("Trying to download the dataset from zenodo...")
                os.makedirs(cifar10c_dir)
                self.download_from_zenodo()
            else:
                raise RuntimeError(f"Please download it manually or set download=True.")

        labels = np.load(os.path.join(cifar10c_dir, 'labels.npy'))
        self.images, self.labels = [], []

        for c in corruptions:  # for mutiple corruption types
            images_all_path = os.path.join(cifar10c_dir, (c + '.npy'))
            images_all = np.load(images_all_path)
            self.images.extend(images_all[(severity - 1) * n_total_cifar: 
                                          severity * n_total_cifar])
            self.labels.extend(labels[(severity - 1) * n_total_cifar: 
                                          severity * n_total_cifar])
    
    def download_from_zenodo(self):
        url = 'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1'
        tar_file = os.path.join(self.root, 'CIFAR-10-C.tar')
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(tar_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
        with tarfile.open(tar_file, 'r:tar') as tar:
            tar.extractall(self.root)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img = self.images[index]
        img = Image.fromarray(img)  # PIL Image
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label


if __name__ == '__main__':
    test_dataset = CIFAR10C(root='data', corruptions=['gaussian_noise', 'shot_noise'], severity=1)

    print(f"Number of images: {len(test_dataset)}")
    print(f"Image shape: {test_dataset[0][0].size}")
    print(f"Label: {test_dataset[0][1]}")
