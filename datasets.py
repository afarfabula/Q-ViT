# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import glob
from io import BytesIO
from bisect import bisect_right

import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import numpy as np
from PIL import Image

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


class ParquetImageNetDataset(torch.utils.data.Dataset):
    """
    Minimal parquet-backed ImageNet dataset for HF-style shards:
    - train-*.parquet
    - validation-*.parquet
    """
    def __init__(self, root, split, transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "pyarrow is required for parquet dataset. Install with: pip install pyarrow"
            ) from exc

        self._pq = pq
        self.files = sorted(glob.glob(os.path.join(root, "data", f"{split}-*.parquet")))
        if not self.files:
            raise RuntimeError(
                f"No parquet files found for split '{split}' under {os.path.join(root, 'data')}"
            )

        self._parquet_files = []
        self._row_groups = []  # (file_index, row_group_index, num_rows)
        cumulative = 0
        self._cumulative_rows = []
        for file_idx, path in enumerate(self.files):
            pf = self._pq.ParquetFile(path)
            self._parquet_files.append(pf)
            for rg_idx in range(pf.num_row_groups):
                num_rows = pf.metadata.row_group(rg_idx).num_rows
                self._row_groups.append((file_idx, rg_idx, num_rows))
                cumulative += num_rows
                self._cumulative_rows.append(cumulative)

        self._cache_key = None
        self._cache_rows = None

    def __len__(self):
        return self._cumulative_rows[-1]

    def _get_row_group(self, group_idx):
        file_idx, rg_idx, _ = self._row_groups[group_idx]
        cache_key = (file_idx, rg_idx)
        if self._cache_key == cache_key and self._cache_rows is not None:
            return self._cache_rows

        table = self._parquet_files[file_idx].read_row_group(rg_idx, columns=["image", "label"])
        rows = table.to_pylist()
        self._cache_key = cache_key
        self._cache_rows = rows
        return rows

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset length {len(self)}")

        group_idx = bisect_right(self._cumulative_rows, index)
        prev_cum = 0 if group_idx == 0 else self._cumulative_rows[group_idx - 1]
        row_idx = index - prev_cum
        row = self._get_row_group(group_idx)[row_idx]

        image_bytes = row["image"]["bytes"]
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label"])
        return image, label


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'IMNET_PARQUET':
        split = 'train' if is_train else 'validation'
        dataset = ParquetImageNetDataset(args.data_path, split=split, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
