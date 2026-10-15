import json
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


class PretrainDataset(Dataset):
    """Images and (optionally) the boxes of their object instances.

    With `ann_file` (a COCO-format json, e.g. COCO's instances_train2017.json or MaskCut pseudo
    boxes), all images listed in the json are used and the boxes of each image are sorted from the
    largest to the smallest instance. Without `ann_file`, all images under `root` are used (only
    the class folders listed in `class_list`, if given) and there are no boxes.

    Each item is `transform(image, boxes)` with boxes a list of (x, y, w, h).
    """

    def __init__(self, root, transform, ann_file=None, class_list=None):
        self.root = root
        self.transform = transform
        if ann_file is not None:
            files, self.offsets, self.boxes = load_coco_boxes(ann_file)
        else:
            files, self.offsets, self.boxes = list_images(root, class_list), None, None
        # a numpy array instead of a list of str: no copy-on-read in the data loader workers
        self.files = np.array(files)

    @property
    def has_boxes(self):
        return self.boxes is not None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.files[index])).convert('RGB')
        boxes = []
        if self.boxes is not None:
            boxes = [tuple(b) for b in self.boxes[self.offsets[index]:self.offsets[index + 1]].tolist()]
        return self.transform(img, boxes)


def load_coco_boxes(ann_file):
    with open(ann_file) as f:
        coco = json.load(f)
    boxes = {img['id']: [] for img in coco['images']}
    for ann in coco['annotations']:
        x, y, w, h = ann['bbox']
        if w >= 1 and h >= 1:
            boxes[ann['image_id']].append((ann.get('area', w * h), x, y, w, h))
    files, offsets, data = [], [0], []
    for img in sorted(coco['images'], key=lambda im: im['id']):
        data += [b[1:] for b in sorted(boxes[img['id']], reverse=True)]  # largest instance first
        files.append(img['file_name'])
        offsets.append(len(data))
    return files, np.asarray(offsets), np.asarray(data, dtype=np.float32).reshape(-1, 4)


def list_images(root, class_list=None):
    keep = None
    if class_list is not None:
        with open(class_list) as f:
            keep = {line.strip() for line in f if line.strip()}
    files = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        rel = os.path.relpath(dirpath, root)
        dirnames.sort()
        if keep is not None and rel == '.':
            dirnames[:] = [d for d in dirnames if d in keep]
            continue
        files += [os.path.normpath(os.path.join(rel, f)) for f in sorted(filenames)
                  if f.lower().endswith(IMG_EXTENSIONS)]
    return files


# ---------------------------------------------------------------------------------------------
# datasets of the linear evaluation (torchvision)
# ---------------------------------------------------------------------------------------------
NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'stl10': 10, 'dtd': 47, 'pets': 37}


def build_linear_dataset(name, root, train, transform, download=False):
    if name == 'cifar10':
        return datasets.CIFAR10(root, train=train, transform=transform, download=download)
    if name == 'cifar100':
        return datasets.CIFAR100(root, train=train, transform=transform, download=download)
    if name == 'stl10':
        return datasets.STL10(root, split='train' if train else 'test', transform=transform,
                              download=download)
    if name == 'dtd':
        return datasets.DTD(root, split='train' if train else 'test', transform=transform,
                            download=download)
    if name == 'pets':
        return datasets.OxfordIIITPet(root, split='trainval' if train else 'test',
                                      transform=transform, download=download)
    raise ValueError(f'unknown dataset {name}')
