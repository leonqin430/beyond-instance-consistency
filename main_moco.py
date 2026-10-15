"""MoCo v2 pre-training with the view configurations of the paper.

Based on the official MoCo code (https://github.com/facebookresearch/moco) and ContrastiveCrop
(https://github.com/xyupeng/ContrastiveCrop).
"""
import argparse
import json
import math
import os
import time
import datetime
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision

from models_moco import MoCo
from util import misc
from util.datasets import PretrainDataset
from views import BOX_MODES, VIEW_MODES, PairedCrop, TwoViewTransform, mismatch_views


def get_args_parser():
    parser = argparse.ArgumentParser('MoCo v2 pre-training', add_help=False)

    # view configuration
    parser.add_argument('--views', default='random', choices=VIEW_MODES,
                        help='how the two views of a positive pair are sampled (see views.py)')
    parser.add_argument('--crop_scale', type=float, nargs=2, default=(0.2, 1.0),
                        help='scale range of the crops (RandomResizedCrop)')
    parser.add_argument('--inst_iou', type=float, default=0.8,
                        help='instance_vs_bg: min. IoU of the instance crop with the instance box')
    parser.add_argument('--bg_iou', type=float, default=0.1,
                        help='instance_vs_bg / only_bg: max. IoU of a background crop with any box')
    parser.add_argument('--max_boxes', type=int, default=10,
                        help='instance_vs_bg: the instance view is taken around one of the '
                             'max_boxes largest instances')

    # data
    parser.add_argument('--data_path', required=True, type=str, help='image folder')
    parser.add_argument('--ann_file', default=None, type=str,
                        help='COCO-format json with the instance boxes (GT or pseudo boxes); '
                             'required by --views instance_vs_bg / only_bg')
    parser.add_argument('--class_list', default=None, type=str,
                        help='without --ann_file: only use these class folders (e.g. ImageNet-100)')

    # model
    parser.add_argument('--arch', default='resnet50', type=str, help='torchvision ResNet')
    parser.add_argument('--moco_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--moco_k', default=65536, type=int, help='queue size')
    parser.add_argument('--moco_m', default=0.999, type=float, help='momentum of the key encoder')
    parser.add_argument('--moco_t', default=0.2, type=float, help='softmax temperature')

    # optimization
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=256, type=int, help='total batch size over all GPUs')
    parser.add_argument('--lr', default=0.03, type=float, help='initial learning rate (cosine schedule)')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    # misc
    parser.add_argument('--output_dir', default='./output_dir', type=str)
    parser.add_argument('--saveckp_freq', default=50, type=int, help='save a checkpoint every x epochs')
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int, help='data loading workers per GPU')
    parser.add_argument('--dist_url', default='env://', type=str)
    return parser


def train_moco(args):
    misc.init_distributed_mode(args)
    misc.fix_random_seeds(args.seed + misc.get_rank())
    print('\n'.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ data ============
    paired_crop = PairedCrop(args.views, size=224, scale=args.crop_scale, inst_iou=args.inst_iou,
                             bg_iou=args.bg_iou, max_boxes=args.max_boxes)
    dataset = PretrainDataset(args.data_path, TwoViewTransform(paired_crop),
                              ann_file=args.ann_file, class_list=args.class_list)
    if args.views in BOX_MODES and not dataset.has_boxes:
        raise ValueError(f'--views {args.views} needs instance boxes, please set --ann_file')
    batch_size_per_gpu = args.batch_size // misc.get_world_size()
    if args.moco_k % (batch_size_per_gpu * misc.get_world_size()) != 0:
        raise ValueError('--moco_k must be a multiple of the total batch size (batch_size_per_gpu x #GPUs)')
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    print(f'Data loaded: there are {len(dataset)} images.')

    # ============ model, loss, optimizer ============
    base_encoder = torchvision.models.__dict__[args.arch]
    model = MoCo(base_encoder, args.moco_dim, args.moco_k, args.moco_m, args.moco_t, mlp=True).cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # ============ optionally resume training ============
    to_restore = {'epoch': 0}
    misc.restart_from_checkpoint(
        os.path.join(args.output_dir, 'checkpoint.pth'),
        run_variables=to_restore,
        state_dict=model.module,
        optimizer=optimizer,
    )
    start_epoch = to_restore['epoch']

    start_time = time.time()
    print('Starting MoCo v2 training!')
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        # cosine learning rate schedule
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_stats = train_one_epoch(model, criterion, data_loader, optimizer, epoch, args)

        save_dict = {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
        }
        misc.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and (epoch + 1) % args.saveckp_freq == 0:
            misc.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch + 1:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if misc.is_main_process():
            with (Path(args.output_dir) / 'log.txt').open('a') as f:
                f.write(json.dumps(log_stats) + '\n')
    total_time = time.time() - start_time
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))


def train_one_epoch(model, criterion, data_loader, optimizer, epoch, args):
    model.train()
    metric_logger = misc.MetricLogger(delimiter='  ')
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for im_q, im_k, valid in metric_logger.log_every(data_loader, args.print_freq, header):
        im_q = im_q.cuda(non_blocking=True)
        im_k = im_k.cuda(non_blocking=True)
        if args.views == 'lower_bound':
            im_q, im_k = mismatch_views([im_q, im_k])

        output, target = model(im_q=im_q, im_k=im_k)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(acc1=(output.argmax(1) == target).float().mean().item() * 100)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        # fraction of the positive pairs that satisfy the constraint of the view configuration
        metric_logger.update(valid_pairs=valid.float().mean().item())
    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MoCo v2 pre-training', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_moco(args)
