# Entry point for joint I-JEPA + ProtoNet pretraining
#
# Usage:
#   python main_joint_pretrain.py --fname configs/gadf_soil_nir_joint.yaml --devices cuda:0

import argparse
import multiprocessing as mp
import pprint
import sys
import yaml

from src.utils.distributed import init_distributed
from src.train_joint import main as app_main

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs/gadf_soil_nir_joint.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')

# -- CLI overrides (applied on top of YAML)
parser.add_argument('--tag', type=str, default=None,
                    help='Override logging.write_tag')
parser.add_argument('--folder', type=str, default=None,
                    help='Override logging.folder')
parser.add_argument('--lr', type=float, default=None,
                    help='Override optimization.lr')
parser.add_argument('--epochs', type=int, default=None,
                    help='Override optimization.epochs')
parser.add_argument('--batch_size', type=int, default=None,
                    help='Override data.batch_size')
parser.add_argument('--warmup', type=int, default=None,
                    help='Override optimization.warmup')
parser.add_argument('--model_name', type=str, default=None,
                    help='Override meta.model_name')
parser.add_argument('--patch_size', type=int, default=None,
                    help='Override mask.patch_size')
parser.add_argument('--wandb_name', type=str, default=None,
                    help='Override wandb.name')
parser.add_argument('--wandb_project', type=str, default=None,
                    help='Override wandb.project')
parser.add_argument('--no_wandb', action='store_true',
                    help='Disable wandb')
parser.add_argument('--probe_freq', type=int, default=None,
                    help='Override probe.freq (0 = disabled)')
parser.add_argument('--lambda_proto', type=float, default=None,
                    help='Override protonet.lambda_proto')
parser.add_argument('--no_proto', action='store_true',
                    help='Disable ProtoNet branch')
parser.add_argument('--meta_batch_size', type=int, default=None,
                    help='Override protonet.meta_batch_size')
parser.add_argument('--proto_warmup', type=int, default=None,
                    help='Override protonet.proto_warmup (epochs to ramp lambda_proto from 0 to target)')
parser.add_argument('--proto_loss_type', type=str, default=None,
                    help='Override protonet.loss_type (e.g. mse, smooth_l1)')
parser.add_argument('--run_id', type=str, default=None,
                    help='Fixed run identifier replacing the auto-generated timestamp in the checkpoint dir')
parser.add_argument('--read_checkpoint', type=str, default=None,
                    help='Absolute path to checkpoint to resume from (also sets load_checkpoint=True)')
parser.add_argument('--start_lr', type=float, default=None,
                    help='Override optimization.start_lr')
parser.add_argument('--final_lr', type=float, default=None,
                    help='Override optimization.final_lr')


def apply_overrides(params, args):
    """Apply CLI overrides on top of YAML params."""
    if args.tag is not None:
        params['logging']['write_tag'] = args.tag
    if args.folder is not None:
        params['logging']['folder'] = args.folder
    if args.run_id is not None:
        params['logging']['run_id'] = args.run_id
    if args.read_checkpoint is not None:
        params['meta']['read_checkpoint'] = args.read_checkpoint
        params['meta']['load_checkpoint'] = True
    if args.lr is not None:
        params['optimization']['lr'] = args.lr
    if args.start_lr is not None:
        params['optimization']['start_lr'] = args.start_lr
    if args.final_lr is not None:
        params['optimization']['final_lr'] = args.final_lr
    if args.epochs is not None:
        params['optimization']['epochs'] = args.epochs
    if args.batch_size is not None:
        params['data']['batch_size'] = args.batch_size
    if args.warmup is not None:
        params['optimization']['warmup'] = args.warmup
    if args.model_name is not None:
        params['meta']['model_name'] = args.model_name
    if args.patch_size is not None:
        params['mask']['patch_size'] = args.patch_size
    if args.no_wandb:
        params.setdefault('wandb', {})['enable'] = False
    if args.wandb_name is not None:
        params.setdefault('wandb', {})['name'] = args.wandb_name
    if args.wandb_project is not None:
        params.setdefault('wandb', {})['project'] = args.wandb_project
    if args.probe_freq is not None:
        params.setdefault('probe', {})['freq'] = args.probe_freq
    if args.lambda_proto is not None:
        params.setdefault('protonet', {})['lambda_proto'] = args.lambda_proto
    if args.no_proto:
        params.setdefault('protonet', {})['enable'] = False
    if args.meta_batch_size is not None:
        params.setdefault('protonet', {})['meta_batch_size'] = args.meta_batch_size
    if args.proto_warmup is not None:
        params.setdefault('protonet', {})['proto_warmup'] = args.proto_warmup
    if args.proto_loss_type is not None:
        params.setdefault('protonet', {})['loss_type'] = args.proto_loss_type
    return params


def process_main(rank, fname, world_size, devices, cli_args):
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

    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    params = apply_overrides(params, cli_args)

    if rank == 0:
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')
    app_main(args=params)


if __name__ == '__main__':
    args = parser.parse_args()

    num_gpus = len(args.devices)
    mp.set_start_method('spawn')

    processes = []
    for rank in range(num_gpus):
        p = mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices, args)
        )
        p.start()
        processes.append(p)

    exit_code = 0
    for p in processes:
        p.join()
        if p.exitcode not in (0, None):
            if exit_code == 0:
                exit_code = p.exitcode if p.exitcode > 0 else 1

    if exit_code != 0:
        sys.exit(exit_code)
