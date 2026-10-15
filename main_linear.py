"""Linear evaluation of a pre-trained backbone (frozen) on CIFAR-10/100, STL-10, DTD and Pets.

MoCo v2 (ResNet-50), as in ContrastiveCrop:
    --arch resnet50 --lr 10.0 --lr_schedule step --lr_steps 60 80
DINO (ViT-S/16), as in DINO (concatenated [CLS] tokens of the last 4 blocks, linear scaling rule):
    --arch vit_small --lr 0.001 --lr_scaling true --lr_schedule cosine --n_last_blocks 4
"""
import argparse
import json
import os
import time
import datetime
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torchvision
from torchvision import transforms

import models_vit
from util import misc
from util.datasets import NUM_CLASSES, build_linear_dataset
from views import IMAGENET_MEAN, IMAGENET_STD


def get_args_parser():
    parser = argparse.ArgumentParser('Linear evaluation', add_help=False)
    parser.add_argument('--pretrained', required=True, type=str, help='checkpoint of main_moco.py / main_dino.py')
    parser.add_argument('--checkpoint_key', default=None, type=str,
                        help='key of the weights in the checkpoint (default: teacher for DINO, state_dict for MoCo)')
    parser.add_argument('--arch', default='resnet50', type=str, help='resnet50 | vit_small | vit_base')
    parser.add_argument('--patch_size', default=16, type=int, help='ViT patch size')
    parser.add_argument('--n_last_blocks', default=4, type=int, help='ViT: concatenate [CLS] of the last n blocks')
    parser.add_argument('--avgpool_patchtokens', default=False, type=misc.bool_flag,
                        help='ViT: also concatenate the average-pooled patch tokens of the last block')

    parser.add_argument('--dataset', default='cifar10', choices=list(NUM_CLASSES))
    parser.add_argument('--data_path', default='./data', type=str, help='root of the torchvision dataset')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=512, type=int, help='total batch size over all GPUs')
    parser.add_argument('--lr', default=10.0, type=float)
    parser.add_argument('--lr_scaling', default=False, type=misc.bool_flag,
                        help='multiply --lr by batch_size / 256 (linear scaling rule of DINO)')
    parser.add_argument('--lr_schedule', default='step', choices=['step', 'cosine'])
    parser.add_argument('--lr_steps', default=[60, 80], type=int, nargs='+', help='epochs of the x0.1 decays (step)')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0., type=float)

    parser.add_argument('--output_dir', default='./output_dir/linear', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dist_url', default='env://', type=str)
    return parser


def build_backbone(args):
    """Frozen backbone and a function that returns its features for the linear classifier."""
    state_dict = misc.load_backbone_state_dict(args.pretrained, args.checkpoint_key)
    if args.arch.startswith('resnet'):
        backbone = torchvision.models.__dict__[args.arch]()
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    else:
        backbone = models_vit.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        feat_dim = backbone.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    msg = backbone.load_state_dict(state_dict, strict=False)
    print(f'Pre-trained weights loaded from {args.pretrained} with msg: {msg}')
    assert len(msg.missing_keys) == 0, 'the checkpoint does not match --arch'
    backbone.cuda().eval().requires_grad_(False)

    def forward_features(x):
        if args.arch.startswith('resnet'):
            return backbone(x)
        out = backbone.get_intermediate_layers(x, args.n_last_blocks)
        feats = torch.cat([o[:, 0] for o in out], dim=-1)
        if args.avgpool_patchtokens:
            feats = torch.cat([feats, out[-1][:, 1:].mean(1)], dim=-1)
        return feats

    return forward_features, feat_dim


def main(args):
    misc.init_distributed_mode(args)
    misc.fix_random_seeds(args.seed + misc.get_rank())
    print('\n'.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ data ============
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    if misc.is_main_process():  # download once
        build_linear_dataset(args.dataset, args.data_path, True, None, download=True)
        build_linear_dataset(args.dataset, args.data_path, False, None, download=True)
    # wait on a CPU group with a long timeout: downloading can take longer than the NCCL timeout
    dist.barrier(group=dist.new_group(backend='gloo', timeout=datetime.timedelta(hours=2)))
    train_set = build_linear_dataset(args.dataset, args.data_path, True, train_transform)
    test_set = build_linear_dataset(args.dataset, args.data_path, False, test_transform)
    batch_size = args.batch_size // misc.get_world_size()
    train_loader = torch.utils.data.DataLoader(
        train_set, sampler=torch.utils.data.DistributedSampler(train_set, shuffle=True),
        batch_size=batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    # every test image is evaluated exactly once
    test_subset = torch.utils.data.Subset(test_set, range(misc.get_rank(), len(test_set), misc.get_world_size()))
    test_loader = torch.utils.data.DataLoader(
        test_subset, batch_size=batch_size, num_workers=args.num_workers, pin_memory=True)
    print(f'Data loaded with {len(train_set)} train and {len(test_set)} test images.')

    # ============ model ============
    forward_features, feat_dim = build_backbone(args)
    linear = nn.Linear(feat_dim, NUM_CLASSES[args.dataset])
    linear.weight.data.normal_(mean=0.0, std=0.01)
    linear.bias.data.zero_()
    linear = nn.parallel.DistributedDataParallel(linear.cuda(), device_ids=[args.gpu])

    lr = args.lr * args.batch_size / 256. if args.lr_scaling else args.lr
    optimizer = torch.optim.SGD(linear.parameters(), lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.lr_schedule == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    criterion = nn.CrossEntropyLoss()

    # ============ training ============
    best_acc, start_time = 0., time.time()
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(forward_features, linear, criterion, train_loader, optimizer, epoch, args)
        scheduler.step()
        test_acc = evaluate(forward_features, linear, test_loader)
        best_acc = max(best_acc, test_acc)
        print(f'Epoch {epoch}: test acc@1 {test_acc:.2f} (best {best_acc:.2f})')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'test_acc1': test_acc, 'best_acc1': best_acc, 'epoch': epoch}
        if misc.is_main_process():
            with (Path(args.output_dir) / 'log.txt').open('a') as f:
                f.write(json.dumps(log_stats) + '\n')
    misc.save_on_master({'linear': linear.module.state_dict(), 'test_acc1': test_acc, 'args': vars(args)},
                        os.path.join(args.output_dir, 'linear.pth'))
    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f'Training time {total_time}. Top-1 accuracy: last {test_acc:.2f}, best {best_acc:.2f}')


def train_one_epoch(forward_features, linear, criterion, data_loader, optimizer, epoch, args):
    linear.train()
    metric_logger = misc.MetricLogger(delimiter='  ')
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for images, target in metric_logger.log_every(data_loader, 20, header):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            feats = forward_features(images)
        output = linear(feats)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(acc1=(output.argmax(1) == target).float().mean().item() * 100)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(forward_features, linear, data_loader):
    linear.eval()
    correct = torch.zeros(2, dtype=torch.float64, device='cuda')  # correct, total
    for images, target in data_loader:
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = linear(forward_features(images))
        correct[0] += (output.argmax(1) == target).sum()
        correct[1] += len(target)
    dist.all_reduce(correct)
    return (correct[0] / correct[1]).item() * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Linear evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
