"""EMD-based similarity between the two views of a positive pair (Sec. 4.3 and Appendix A.3).

Each view is split into N patches (grid-based or sampling-based, Fig. 3), and every patch is
encoded by a frozen pre-trained encoder into one local feature. With the cost matrix
C_ij = 1 - cos(x_i, y_j) between the local features of the two views and the optimal transport
plan P* (Earth Mover's Distance), the similarity of the two views is (Algorithm 1)

    S(X, Y) = <P*, 1 - C>.

Following DeepEMD (Zhang et al., CVPR 2020, https://github.com/icoz69/DeepEMD, MIT License), the
local features are centered and the marginal weights s, d come from the cross-reference mechanism;
s and d are normalized to sum to 1, so that S is a weighted average of cosine similarities in
[-1, 1]. A higher S means that the two views share more information (lower view diversity).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as TF

# encoders of the paper (Appendix B.5) and the input size of their patches
ENCODERS = {
    'resnet50': 84,        # ResNet-50 (EMD of the MoCo v2 results)
    'resnet101': 84,       # ResNet-101
    'vit_small': 224,      # ViT-S/16 (EMD of the DINO results)
    'vit_base': 224,       # ViT-B/16
    'dino_vits16': 224,    # ViT-DINO-S/16
    'dino_vitb16': 224,    # ViT-DINO-B/16
    'dinov2_vits14': 224,  # DINOv2 ViT-S/14
    'dinov2_vitb14': 224,  # DINOv2 ViT-B/14
}


def build_encoder(name):
    """Frozen pre-trained encoder that maps a batch of patches to one feature per patch."""
    if name in ('resnet50', 'resnet101'):  # supervised ImageNet weights of torchvision
        model = getattr(torchvision.models, name)(weights='IMAGENET1K_V1')
        model.fc = nn.Identity()  # global average pooled features
    elif name in ('vit_small', 'vit_base'):  # supervised ImageNet ViTs (DeiT), needs timm
        import timm
        model = timm.create_model(f'deit_{name[4:]}_patch16_224', pretrained=True, num_classes=0)
    elif name.startswith('dinov2'):
        model = torch.hub.load('facebookresearch/dinov2', name)  # [CLS] token
    elif name.startswith('dino'):
        model = torch.hub.load('facebookresearch/dino:main', name)  # [CLS] token
    else:
        raise ValueError(f'unknown encoder {name}')
    return model.eval().requires_grad_(False)


def _grid_cells(length, num_grid, ratio):
    # cells of a num_grid x num_grid grid, enlarged by `ratio` around their centers (DeepEMD)
    cell, enlarged = int(length / num_grid), int(length / num_grid * ratio)
    centers = [cell // 2 + k * cell for k in range(num_grid)]
    return [(max(0, c - enlarged // 2), min(length, c + enlarged // 2)) for c in centers]


def grid_patches(views, grids=(2, 3), ratio=2., size=84):
    """Grid-based patches: [B, 3, H, W] -> [B, sum(g * g for g in grids), 3, size, size]."""
    H, W = views.shape[-2:]
    patches = []
    for g in grids:
        for top, bottom in _grid_cells(H, g, ratio):
            for left, right in _grid_cells(W, g, ratio):
                patches.append(TF.resized_crop(views, top, left, bottom - top, right - left,
                                               [size, size], antialias=True))
    return torch.stack(patches, dim=1)


def sampled_patches(views, num_patches=9, size=84, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
    """Sampling-based patches (random resized crops): [B, 3, H, W] -> [B, num_patches, 3, size, size]."""
    patches = []
    for view in views:
        for _ in range(num_patches):
            top, left, h, w = transforms.RandomResizedCrop.get_params(view, scale, ratio)
            patches.append(TF.resized_crop(view, top, left, h, w, [size, size], antialias=True))
    return torch.stack(patches).view(len(views), num_patches, *patches[0].shape)


def _emd_opencv(cost, weight1, weight2):
    # exact EMD (OpenCV), one transport problem at a time on CPU
    import cv2
    cost = cost.detach().cpu().numpy().astype(np.float32)
    weight1 = weight1.detach().cpu().numpy().astype(np.float32)
    weight2 = weight2.detach().cpu().numpy().astype(np.float32)
    plans = [cv2.EMD(w1.reshape(-1, 1), w2.reshape(-1, 1), cv2.DIST_USER, c)[2]
             for c, w1, w2 in zip(cost, weight1, weight2)]
    return torch.from_numpy(np.stack(plans))


def _emd_sinkhorn(cost, weight1, weight2, lam=20., iters=10):
    # Sinkhorn-Knopp iterations for argmin_P <P, C> - h(P) / lam (Algorithm 1), in the log domain,
    # batched on GPU
    log_w1, log_w2 = weight1.log(), weight2.log()
    f, g = torch.zeros_like(weight1), torch.zeros_like(weight2)
    for _ in range(iters):
        f = (log_w1 - torch.logsumexp(lam * (g[:, None, :] - cost), dim=2)) / lam
        g = (log_w2 - torch.logsumexp(lam * (f[:, :, None] - cost), dim=1)) / lam
    return torch.exp(lam * (f[:, :, None] + g[:, None, :] - cost))


def emd_similarity(x, y, solver='opencv', lam=20., iters=10):
    """EMD-based similarity of two sets of local features x: [B, N, D] and y: [B, M, D] -> [B]."""
    x, y = x.float(), y.float()
    # marginal weights: relevance of every local feature to the other view (cross-reference, DeepEMD)
    weight1 = F.relu((x * y.mean(1, keepdim=True)).sum(-1)) + 1e-3
    weight2 = F.relu((y * x.mean(1, keepdim=True)).sum(-1)) + 1e-3
    weight1 = weight1 / weight1.sum(1, keepdim=True)
    weight2 = weight2 / weight2.sum(1, keepdim=True)
    # cosine similarity 1 - C of the centered local features
    x = F.normalize(x - x.mean(-1, keepdim=True), dim=-1)
    y = F.normalize(y - y.mean(-1, keepdim=True), dim=-1)
    sim = x @ y.transpose(1, 2)
    if solver == 'opencv':
        plan = _emd_opencv(1 - sim, weight1, weight2).to(sim.device)
    elif solver == 'sinkhorn':
        plan = _emd_sinkhorn(1 - sim, weight1, weight2, lam, iters)
    else:
        raise ValueError(f'unknown solver {solver}')
    return (plan * sim).sum((1, 2))


@torch.no_grad()
def view_similarity(encoder, view1, view2, strategy='grid', patch_size=84, solver='opencv',
                    chunk_size=512, **kwargs):
    """EMD-based similarity of a batch of positive pairs (view1[i], view2[i]) -> [B]."""
    views = torch.cat([view1, view2])
    if strategy == 'grid':
        patches = grid_patches(views, size=patch_size)
    elif strategy == 'sampling':
        patches = sampled_patches(views, size=patch_size)
    else:
        raise ValueError(f'unknown strategy {strategy}')
    num_views, num_patches = patches.shape[:2]
    feats = torch.cat([encoder(p) for p in patches.flatten(0, 1).split(chunk_size)])
    feats = feats.view(num_views, num_patches, -1)
    return emd_similarity(feats[:len(view1)], feats[len(view1):], solver=solver, **kwargs)
