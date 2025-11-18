# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import multiprocessing as mp

import pprint
import yaml

from src.utils.distributed import init_distributed
from src.train import main as app_main

import torch

from src.utils.wandb_utils import init_wandb, wandb

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine; use "all" to select every available GPU')


def process_main(rank, fname, world_size, devices):
    
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    # -- load script params
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- initialize W&B (only rank 0 will actually create a run)
    wandb_run = init_wandb(params, rank)

    app_main(args=params)


if __name__ == '__main__':
    args = parser.parse_args()

    if 'all' in args.devices:
        device_count = torch.cuda.device_count()
        args.devices = [f'cuda:{i}' for i in range(device_count)]
        print(f"Auto-detected {device_count} GPUs: {args.devices}")

    num_gpus = len(args.devices)
    mp.set_start_method('spawn')

    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices)
        ).start()
