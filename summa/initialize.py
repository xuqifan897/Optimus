import random
import os

import numpy as np
import torch

from summa import get_adlr_autoresume
from summa import get_args
from summa import get_tensorboard_writer
from summa import mpu
from summa.global_vars import set_global_variables
from summa.mpu import set_model_parallel_rank, set_model_parallel_world_size

def initialize_optimus(extra_args_provider=None, args_defaults={},
                        ignore_unknown_args=False, allow_no_cuda=False):
    """Set global variables, initialize distributed, and
        set autoresume and random seeds.
        `allow_no_cuda` should not be set unless using optimus for cpu only
        data processing. In general this arg should not be set unless you know
        what you are doing.
        Returns a function to finalize distributed env initialization
        (optionally, only when args.lazy_mpu_init == True)

    """
    if not allow_no_cuda:
        # Make sure cuda is available.
        assert torch.cuda.is_available(), 'Megatron requires CUDA, ' \
                                          'rank {}'.format(int(os.getenv('SLURM_PROCID', '0')))

    # Parse args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(extra_args_provider=extra_args_provider,
                         args_defaults=args_defaults,
                         ignore_unknown_args=ignore_unknown_args)

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed()

        # Random seeds for reproducibility.
        if args.rank == 0:
            print('> setting random seeds to {} ...'.format(args.seed))
        _set_random_seed(args.seed)

    args = get_args()

    if args.lazy_mpu_init:
        args.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        set_model_parallel_world_size(args.model_parallel_size)
        # and return function for external DDP manager to call when it has DDP initialized
        set_model_parallel_rank(args.rank)
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Initialize memory buffers.
        _initialize_mem_buffs()

        # Autoresume.
        _init_autoresume()

        # Write arguments to tensorboard.
        _write_args_to_tensorboard()
        # No continuation function
        return None


def _initialize_distributed():
    """Initialize torch.distributed and mpu."""
    args = get_args()

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():

        if args.rank == 0:
            print('torch distributed is already initialized, '
                  'skipping initialization ...', flush=True)
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    else:

        if args.rank == 0:
            print('> initializing torch distributed ...', flush=True)
        # Manually set the device ids.
        if device_count > 0:
            device = args.rank_original % device_count
            if args.local_rank is not None:
                assert args.local_rank == device, \
                    'expected local-rank to be the same as rank % device-count.'
            else:
                args.local_rank = device
            torch.cuda.set_device(device)

        torch.distributed.init_process_group(
            backend=args.distributed_backend,
            world_size=args.world_size, rank=args.rank,
            init_method=args.init_method)

    # Set the model-parallel / data-parallel communicators.
    if device_count > 0:
        mpu.initialize_model_parallel(args.model_parallel_size, args.summa_dim)


def _init_autoresume():
    """Set autoresume start time."""
    autoresume = get_adlr_autoresume()
    if autoresume:
        torch.distributed.barrier()
        autoresume.init()
        torch.distributed.barrier()


def _set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            mpu.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError('Seed ({}) should be a positive integer.'.format(seed))


def _write_args_to_tensorboard():
    """Write arguments to tensorboard."""
    args = get_args()
    writer = get_tensorboard_writer()
    if writer:
        for arg in vars(args):
            writer.add_text(arg, str(getattr(args, arg)))


def _initialize_mem_buffs():
    """Initialize manually allocated static memory."""
    args = get_args()

    # Initialize memory for checkpointed activations.
    if args.distribute_checkpointed_activations:
        mpu.init_checkpointed_activations_memory_buffer()
        mpu.init_workspace_memory_buffer()
        # mpu.init_forward_buffer()
        mpu.init_QKV_forward_buffer()
        mpu.init_QKV_dense_buffer()
        mpu.init_h4h_forward_buffer()
        mpu.init_fhh_forward_buffer()
        mpu.init_backward_buffer()
        mpu.init_parameter_gradient_buffer()
        mpu.init_conjunction_gradient_buffer()
        # if not args.ParallelTransformer_only:
        #     mpu.init_lmhead_dense_buffer()