"""EMD-based view similarity of a view configuration (Sec. 4.3).

The positive pairs are generated exactly as for pre-training (same view configuration and MoCo v2
augmentations) and the EMD-based similarity of every pair is averaged over (a fraction of) the
dataset. Runs on one GPU (`python main_emd.py ...`) or on several (`torchrun ... main_emd.py ...`).
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from emd import ENCODERS, build_encoder, view_similarity
from util import misc
from util.datasets import PretrainDataset
from views import BOX_MODES, VIEW_MODES, PairedCrop, TwoViewTransform, mismatch_views


def get_args_parser():
    parser = argparse.ArgumentParser('EMD-based view similarity', add_help=False)

    # view configuration (same as main_moco.py)
    parser.add_argument('--views', default='random', choices=VIEW_MODES)
    parser.add_argument('--crop_scale', type=float, nargs=2, default=(0.2, 1.0))
    parser.add_argument('--inst_iou', type=float, default=0.8)
    parser.add_argument('--bg_iou', type=float, default=0.1)
    parser.add_argument('--max_boxes', type=int, default=10)

    # data
    parser.add_argument('--data_path', required=True, type=str, help='image folder')
    parser.add_argument('--ann_file', default=None, type=str, help='COCO-format json with the instance boxes')
    parser.add_argument('--class_list', default=None, type=str, help='only use these class folders')
    parser.add_argument('--data_fraction', default=1.0, type=float,
                        help='fraction of the images to use (0.001 already gives stable scores)')
    parser.add_argument('--batch_size_per_gpu', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    # EMD
    parser.add_argument('--encoder', default='resnet50', choices=list(ENCODERS),
                        help='frozen pre-trained encoder of the patches')
    parser.add_argument('--patch_size', default=None, type=int,
                        help='input size of the patches (default: 84 for ResNets, 224 for ViTs)')
    parser.add_argument('--strategy', default=['grid', 'sampling'], nargs='+', choices=['grid', 'sampling'],
                        help='how the local features of a view are obtained (Fig. 3)')
    parser.add_argument('--solver', default='opencv', choices=['opencv', 'sinkhorn'],
                        help='exact EMD (OpenCV, CPU) or Sinkhorn-Knopp iterations (Algorithm 1, GPU)')
    parser.add_argument('--sinkhorn_lambda', default=20., type=float,
                        help='Sinkhorn: weight lambda of the transport cost (entropy weight 1 / lambda)')
    parser.add_argument('--sinkhorn_iters', default=10, type=int, help='Sinkhorn: number of iterations T')

    # misc
    parser.add_argument('--output_dir', default=None, type=str, help='save the scores to output_dir/emd.json')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dist_url', default='env://', type=str)
    return parser


def main(args):
    misc.init_distributed_mode(args)
    misc.fix_random_seeds(args.seed + misc.get_rank())
    print('\n'.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    patch_size = args.patch_size or ENCODERS[args.encoder]

    # ============ data ============
    paired_crop = PairedCrop(args.views, size=224, scale=args.crop_scale, inst_iou=args.inst_iou,
                             bg_iou=args.bg_iou, max_boxes=args.max_boxes)
    dataset = PretrainDataset(args.data_path, TwoViewTransform(paired_crop),
                              ann_file=args.ann_file, class_list=args.class_list)
    if args.views in BOX_MODES and not dataset.has_boxes:
        raise ValueError(f'--views {args.views} needs instance boxes, please set --ann_file')
    # the same random subset on every process, split without padding
    generator = torch.Generator().manual_seed(args.seed)
    num_images = max(int(len(dataset) * args.data_fraction), misc.get_world_size())
    indices = torch.randperm(len(dataset), generator=generator)[:num_images].tolist()
    subset = torch.utils.data.Subset(dataset, indices[misc.get_rank()::misc.get_world_size()])
    data_loader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f'Computing the EMD-based similarity of {num_images} / {len(dataset)} images.')

    # ============ EMD ============
    if not misc.is_main_process():
        dist.barrier()  # the main process downloads the weights first
    encoder = build_encoder(args.encoder).cuda()
    if misc.is_main_process():
        dist.barrier()
    stats = torch.zeros(len(args.strategy) + 2, dtype=torch.float64, device='cuda')  # sums, #pairs, #valid
    metric_logger = misc.MetricLogger(delimiter='  ')
    for view1, view2, valid in metric_logger.log_every(data_loader, 10, 'EMD:'):
        if args.views == 'lower_bound' and len(view1) < 2:
            continue  # a view needs a partner from another image
        view1 = view1.cuda(non_blocking=True)
        view2 = view2.cuda(non_blocking=True)
        if args.views == 'lower_bound':
            view1, view2 = mismatch_views([view1, view2])
        for i, strategy in enumerate(args.strategy):
            score = view_similarity(encoder, view1, view2, strategy, patch_size, args.solver,
                                    lam=args.sinkhorn_lambda, iters=args.sinkhorn_iters)
            stats[i] += score.sum()
            metric_logger.update(**{strategy: score.mean().item()})
        stats[-2] += len(view1)
        stats[-1] += valid.sum()
    dist.all_reduce(stats)
    if stats[-2] == 0:
        raise ValueError('no positive pairs, please increase --data_fraction')

    results = {f'emd_{s}': (stats[i] / stats[-2]).item() for i, s in enumerate(args.strategy)}
    results.update(num_pairs=int(stats[-2].item()), valid_pairs=(stats[-1] / stats[-2]).item())
    print(json.dumps(results))
    if args.output_dir is not None and misc.is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(args.output_dir, 'emd.json'), 'w') as f:
            json.dump({**results, 'args': vars(args)}, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('EMD-based view similarity', parents=[get_args_parser()])
    main(parser.parse_args())
