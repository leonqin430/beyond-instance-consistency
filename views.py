"""View configurations of the paper (Sec. 4.1, 4.2 and Fig. 2).

The two views of a positive pair are crops of one image whose boxes are sampled *jointly* under a
spatial constraint. Each crop then goes through the usual augmentations of the SSL method.

    random          two independent RandomResizedCrops (Baseline). Combined with --crop_scale
                    this also gives Smaller Crop and Larger Crop.
    zero_overlap    the two crops do not overlap (Zero Spatial Overlap). Combined with a smaller
                    --crop_scale this gives Smaller Crop + Zero Spatial Overlap.
    instance_vs_bg  one crop covers an object instance (IoU > inst_iou with its box), the other
                    one only covers background (IoU < bg_iou with every box); no overlap.
    only_bg         both crops only cover background (IoU < bg_iou with every box); no overlap.
    lower_bound     two independent crops, but the two views of a positive pair are taken from
                    different images (done at batch level with `mismatch_views`).

Instance boxes are ground-truth boxes (COCO) or pseudo boxes of an unsupervised method such as
MaskCut (ImageNet-100). All boxes are (x, y, w, h) in pixels of the original image. The order of
the two views of a pair is random.
"""
import random

from PIL import ImageFilter
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

VIEW_MODES = ('random', 'zero_overlap', 'instance_vs_bg', 'only_bg', 'lower_bound')
BOX_MODES = ('instance_vs_bg', 'only_bg')  # modes that need instance boxes

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def box_iou(a, b):
    """IoU of two (x, y, w, h) boxes."""
    iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    inter = max(iw, 0) * max(ih, 0)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.


def enlarge_box(box, margin, width, height):
    """Enlarge a box by `margin` of its size on every side, clipped to the image."""
    x, y, w, h = box
    x0, y0 = max(0., x - margin * w), max(0., y - margin * h)
    x1, y1 = min(float(width), x + w + margin * w), min(float(height), y + h + margin * h)
    return x0, y0, max(x1 - x0, 1.), max(y1 - y0, 1.)


class PairedCrop:
    """Sample the two crops of a positive pair under the constraint of a view configuration.

    Args:
        mode: one of VIEW_MODES.
        size: output size of each crop.
        scale, ratio: area and aspect-ratio ranges of the crops, as in RandomResizedCrop.
        inst_iou: 'instance_vs_bg': the instance crop has IoU > inst_iou with the instance box.
        bg_iou: 'instance_vs_bg', 'only_bg': a background crop has IoU < bg_iou with every box.
        max_boxes: 'instance_vs_bg': the instance crop is taken around one of the `max_boxes`
            largest instances (the background constraint always uses all instances).
        max_tries: maximum number of crop proposals per view (rejection sampling).

    Returns (crop1, crop2, valid); valid is False if the constraint could not be met, in which
    case two unconstrained crops are returned.
    """

    def __init__(self, mode='random', size=224, scale=(0.2, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation=InterpolationMode.BILINEAR, inst_iou=0.8, bg_iou=0.1,
                 max_boxes=10, max_tries=1000):
        assert mode in VIEW_MODES, f'unknown view mode {mode}'
        self.mode = mode
        self.size = (size, size)
        self.scale, self.ratio = tuple(scale), tuple(ratio)
        self.interpolation = interpolation
        self.inst_iou, self.bg_iou = inst_iou, bg_iou
        self.max_boxes, self.max_tries = max_boxes, max_tries

    def __call__(self, img, boxes=()):
        box1, box2, valid = self.sample_boxes(img, boxes)
        return self.crop(img, box1), self.crop(img, box2), valid

    def sample_boxes(self, img, boxes=()):
        boxes = list(boxes)  # sorted from the largest to the smallest instance
        if self.mode in ('random', 'lower_bound'):
            pair = self._random_box(img), self._random_box(img)
        elif self.mode == 'zero_overlap':
            pair = self._pair(img, lambda b: True, lambda b1, b2: box_iou(b1, b2) == 0)
        elif self.mode == 'only_bg':
            pair = self._pair(img, lambda b: self._is_bg(b, boxes),
                              lambda b1, b2: self._is_bg(b2, boxes) and box_iou(b1, b2) == 0)
        else:
            pair = self._instance_vs_bg(img, boxes)
        if pair is None:
            return self._random_box(img), self._random_box(img), False
        if random.random() < 0.5:  # box2 is sampled given box1: randomize the order of the views
            pair = pair[::-1]
        return pair[0], pair[1], True

    def crop(self, img, box):
        # round the edges (not x and w) so that non-overlapping boxes stay non-overlapping
        x0, y0 = round(box[0]), round(box[1])
        x1, y1 = max(round(box[0] + box[2]), x0 + 1), max(round(box[1] + box[3]), y0 + 1)
        return F.resized_crop(img, y0, x0, y1 - y0, x1 - x0, self.size, self.interpolation,
                              antialias=True)

    def _random_box(self, img):
        top, left, h, w = transforms.RandomResizedCrop.get_params(img, self.scale, self.ratio)
        return left, top, w, h

    def _sample_where(self, img, cond, tries):
        for _ in range(tries):
            box = self._random_box(img)
            if cond(box):
                return box
        return None

    def _pair(self, img, cond1, cond2, restarts=10):
        # sample box1 with cond1(box1), then box2 with cond2(box1, box2); a new box1 is drawn if no
        # box2 is found (e.g. box1 covers most of the image)
        tries = max(self.max_tries // restarts, 1)
        for _ in range(restarts):
            box1 = self._sample_where(img, cond1, tries)
            if box1 is None:
                continue
            box2 = self._sample_where(img, lambda b: cond2(box1, b), tries)
            if box2 is not None:
                return box1, box2
        return None

    def _is_bg(self, box, boxes):
        return all(box_iou(box, b) < self.bg_iou for b in boxes)

    def _instance_vs_bg(self, img, boxes):
        if len(boxes) == 0:
            return None
        inst = random.choice(boxes[:self.max_boxes])
        box1 = None
        if self._reachable(img, inst):
            box1 = self._sample_where(img, lambda b: box_iou(b, inst) > self.inst_iou, self.max_tries)
        if box1 is None:
            # e.g. instances too small for the crop scale: crop the instance box, enlarged by a
            # margin that keeps IoU > inst_iou ((1 + 2 * margin) ** 2 < 1 / inst_iou)
            box1 = enlarge_box(inst, 0.45 * (self.inst_iou ** -0.5 - 1), *img.size)
        box2 = self._sample_where(img, lambda b: self._is_bg(b, boxes) and box_iou(b, box1) == 0,
                                  self.max_tries)
        return None if box2 is None else (box1, box2)

    def _reachable(self, img, inst):
        # IoU(crop, inst) <= min(areas) / max(areas): can a crop of the given scale reach inst_iou?
        img_area, inst_area = img.size[0] * img.size[1], inst[2] * inst[3]
        return (inst_area > self.inst_iou * self.scale[0] * img_area
                and self.inst_iou * inst_area < self.scale[1] * img_area)


def mismatch_views(views):
    """'lower_bound': view k of every sample is replaced by view k of another image of the batch
    (the batch is rolled by k), so that no two views of a "positive pair" share an image."""
    return [v.roll(k, dims=0) for k, v in enumerate(views)]


class GaussianBlur:
    """Gaussian blur of SimCLR / MoCo v2 / DINO."""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.sigma)))


class TwoViewTransform:
    """MoCo v2 augmentation of the two crops sampled by `PairedCrop`."""

    def __init__(self, paired_crop, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.paired_crop = paired_crop
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __call__(self, img, boxes=()):
        crop1, crop2, valid = self.paired_crop(img, boxes)
        return self.augment(crop1), self.augment(crop2), valid


class MultiCropTransform:
    """DINO multi-crop augmentation; the two global crops are sampled by `PairedCrop`, the local
    crops are sampled as in DINO."""

    def __init__(self, paired_crop, local_crops_number=8, local_crops_scale=(0.05, 0.4),
                 local_crops_size=96, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        self.paired_crop = paired_crop
        self.global_transfo1 = transforms.Compose([
            flip_and_color_jitter,
            GaussianBlur(),
            normalize,
        ])
        self.global_transfo2 = transforms.Compose([
            flip_and_color_jitter,
            transforms.RandomApply([GaussianBlur()], p=0.1),
            transforms.RandomSolarize(128, p=0.2),
            normalize,
        ])
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(local_crops_size, scale=local_crops_scale,
                                         interpolation=InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            transforms.RandomApply([GaussianBlur()], p=0.5),
            normalize,
        ])

    def __call__(self, img, boxes=()):
        crop1, crop2, valid = self.paired_crop(img, boxes)
        crops = [self.global_transfo1(crop1), self.global_transfo2(crop2)]
        crops += [self.local_transfo(img) for _ in range(self.local_crops_number)]
        return crops, valid
