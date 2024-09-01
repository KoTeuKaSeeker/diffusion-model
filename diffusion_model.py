import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import List

import torch
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms

class CatDataset(IterableDataset):
    def __init__(self, path: str, shard_size: int, batch_size: int, target_image_size: tuple, 
                 rank: int, world_size: int, shuffle: bool = False, allows_ex: List[str] = [".jpg", ".png", ".JPEG"]):
        self.shard_size = shard_size
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size

        self.image_paths = [os.path.join(path, name) for name in os.listdir(path) if os.path.splitext(name)[1] in allows_ex]

        self.count_images = len(self.image_paths)
        self.count_shards = self.count_images // shard_size

        self.transform = transforms.Compose([
            transforms.Resize(target_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.unnormalize = transforms.Compose([
            transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        ])

        if shuffle:
            np.random.shuffle(self.image_paths)


    def get_shard(self, shard_n) -> torch.Tensor:
        """
        Loads a shard of images from a hard drive and then preprocesses it
        """
        assert shard_n < self.count_shards, "shard_n has a greater value than count_shards."
        shard_start_id = shard_n * self.shard_size
        shard_image_paths = self.image_paths[shard_start_id:shard_start_id + self.shard_size]
        
        images = []
        for path in shard_image_paths:
            image = Image.open(path).convert('RGB')
            image = self.transform(image)
            images.append(image)
        
        shard_images = torch.stack(images, 0)
        return shard_images


    def __iter__(self):
        for shard_id in range(self.count_shards):
            shard_images = self.get_shard(shard_id)
            image_id = self.rank * self.batch_size
            while image_id + self.batch_size <= self.shard_size:
                images_batch = shard_images[image_id:image_id + self.batch_size]
                yield images_batch
                image_id += self.world_size * self.batch_size


    def __len__(self) -> int:
        return self.count_images


def clean_folder(path: str, allows_ex: List[str]):
    """
        Filters the folder by allowed extensions
    """
    for name in os.listdir(path):
        _, ex = os.path.splitext(name)
        if ex not in allows_ex:
            file_path = os.path.join(path, name)
            os.remove(file_path)

train_dataset = CatDataset(path=r"data\images", shard_size=1024, batch_size=16, target_image_size=(100, 100), rank=0, world_size=1)
train_iterator = iter(train_dataset)

for x in train_iterator:
    for image in x:
        image = train_dataset.unnormalize(image)
        image = transforms.ToPILImage()(image)
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    
    