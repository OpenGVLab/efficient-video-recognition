#!/usr/bin/env python

import argparse
from typing import Dict

import torch
import torch.distributed as dist

from .dataset import VideoDataset, DummyDataset

def setup_arg_parser(parser: argparse.ArgumentParser):
    parser.add_argument('--train_list_path', type=str,
                        help='path to training data list')
    parser.add_argument('--val_list_path', type=str,
                        help='path to validation data list')
    parser.add_argument('--train_data_root', type=str,
                        help='training samples root directory')
    parser.add_argument('--val_data_root', type=str,
                        help='validation samples root directory')
    parser.add_argument('--data_root', type=str, default='',
                        help='training and validation samples root directory, might be overrided by --train_data_root or --val_data_root')

    parser.add_argument('--batch_size', type=int,
                        help='training batch size on a all GPUs')

    parser.add_argument('--num_spatial_views', type=int, default=1,
                        help='number of spatial crops used for testing (total views = num_spatial_views * num_temporal_views)')
    parser.add_argument('--num_temporal_views', type=int, default=3,
                        help='number of temporal crops used for testing (total views = num_spatial_views * num_temporal_views)')
    parser.add_argument('--num_frames', type=int, default=8,
                        help='number of frames used for each view')
    parser.add_argument('--sampling_rate', type=int, default=16,
                        help='temporal stride for frame sampling, only valid when tsn_sampling is not enabled')
    parser.add_argument('--tsn_sampling', action='store_true',
                        help='enable TSN-style sampling (i.e. sample frames with dynamic stride to cover the whole video)')
    parser.add_argument('--spatial_size', type=int, default=224,
                        help='frame height and width in pixels')

    parser.add_argument('--mean', type=float, nargs='+',
                        help='pixel mean used to normalize the image.')
    parser.add_argument('--std', type=float, nargs='+',
                        help='pixel std used to normalize the image')

    parser.add_argument('--num_workers', type=int, default=10,
                        help='number of DataLoader worker threads')
    
    parser.add_argument('--dummy_dataset', action='store_true',
                        help='use fake datasets that generate all 0 (use for speed test only)')

    parser.add_argument('--auto_augment', type=str,
                        help='auto augment configuration')
    parser.add_argument('--interpolation', type=str, default='bicubic',
                        help='interpolation mode')
    parser.add_argument('--no_mirror', action='store_false', dest='mirror',
                        help='disable mirror for training (frequently used for the something-something dataset)')
    parser.set_defaults(mirror=True)
                        

def _parse_mean_and_std(args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    def parse_mean_or_std(arg, default_value):
        if arg is None:
            return torch.Tensor([default_value] * 3)
        elif len(arg) == 1:
            return torch.Tensor(arg * 3)
        elif len(arg) == 3:
            return torch.Tensor(arg)
        else:
            raise NotImplementedError()
    return {
        'mean': parse_mean_or_std(args.mean, 0.45),
        'std': parse_mean_or_std(args.std, 0.225),
    }


def create_train_dataset(args: argparse.Namespace) -> torch.utils.data.Dataset:
    if args.dummy_dataset:
        return DummyDataset(
            list_path=args.train_list_path,
            num_frames=args.num_frames,
            num_views=1,
            spatial_size=args.spatial_size,
        )

    return VideoDataset(
        list_path=args.train_list_path,
        data_root=args.train_data_root or args.data_root,
        num_spatial_views=1, num_temporal_views=1, random_sample=True,
        auto_augment=args.auto_augment,
        interpolation=args.interpolation,
        mirror=args.mirror,
        num_frames=args.num_frames,
        sampling_rate=-1 if args.tsn_sampling else args.sampling_rate,
        spatial_size=args.spatial_size,
        **_parse_mean_and_std(args),
    )


def create_train_loader(args: argparse.Namespace, resume_step: int = 0) -> torch.utils.data.DataLoader:
    dataset = create_train_dataset(args)
    rank, world_size = (0, 1) if not dist.is_initialized() else (dist.get_rank(), dist.get_world_size())

    assert args.batch_size % world_size == 0
    batch_size_per_gpu = args.batch_size // world_size

    # manually create a step-based sampler
    sampler = []
    while len(sampler) * len(dataset) < args.num_steps * args.batch_size:
        g = torch.Generator()
        g.manual_seed(len(sampler))
        indices = torch.randperm(len(dataset), generator=g)
        sampler.append(indices)
    sampler = torch.cat(sampler, dim=0)[:args.num_steps * args.batch_size].view(args.num_steps, args.batch_size)
    sampler = sampler[resume_step:, batch_size_per_gpu * rank: batch_size_per_gpu * (rank + 1)].flatten().tolist()

    loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler, batch_size=batch_size_per_gpu,
        num_workers=args.num_workers, pin_memory=False, drop_last=True,
    )

    return loader


def create_val_dataset(args: argparse.Namespace) -> torch.utils.data.Dataset:
    if args.dummy_dataset:
        return DummyDataset(
            list_path=args.val_list_path,
            num_frames=args.num_frames,
            num_views=args.num_spatial_views * args.num_temporal_views,
            spatial_size=args.spatial_size,
        )

    return VideoDataset(
        list_path=args.val_list_path,
        data_root=args.val_data_root or args.data_root,
        num_spatial_views=args.num_spatial_views,
        num_temporal_views=args.num_temporal_views,
        random_sample=False,
        num_frames=args.num_frames,
        sampling_rate=-1 if args.tsn_sampling else args.sampling_rate,
        spatial_size=args.spatial_size,
        **_parse_mean_and_std(args),
    )


def create_val_loader(args: argparse.Namespace) -> torch.utils.data.Dataset:
    dataset = create_val_dataset(args)
    rank, world_size = (0, 1) if not dist.is_initialized() else (dist.get_rank(), dist.get_world_size())

    # sampler for distribued eval
    sampler = list(range(rank, len(dataset), world_size))

    loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler, batch_size=1,
        num_workers=args.num_workers, pin_memory=False,
    )

    return loader
