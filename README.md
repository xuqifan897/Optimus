# Optimus
This is the code for paper "An Efficient 2D Method for Training Super-Large Deep Learning Models" (https://arxiv.org/abs/2104.05343)

Requirements:

pybind11

torch 1.5.0

six

regex

The code is tested on TACC Frontera, a SLURM system. Some modifications are needed to run on a normal ubuntu system (ubuntu, for simplicity).
To test the benchmark code, please run: bash bcmk_ParallelTransformer.sh. On SLURM, processes are spawn with the built in command srun. On ubuntu, users can either use torch.distributed.launch command (in https://pytorch.org/docs/stable/distributed.html) or mpirun or mpiexec.

A full list of arguments is provided in summa/arguments.py. Please note os.getenv() function may have different environment variables from ubuntu. In our implementation, rank=int(os.getenv('SLURM_PROCID', '0')). For torch.distributed, the system would pass the rank to args.local_rank. For other methods, please revise the code accordingly.
args.world_size and master_addr are also from os.getenv() function. args.init_method is the input argument for torch.distributed.init_process_group(). Please revise it accordingly.

The full training code will be available soon.
