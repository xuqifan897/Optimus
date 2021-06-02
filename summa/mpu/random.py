# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch

import contextlib

import torch
from torch import _C
from torch.cuda import _lazy_call, device as device_ctx_manager
from torch.utils.checkpoint import detach_variable

from summa import get_args
from summa.memory import allocate_mem_buff
from summa.memory import SimpleBuffer

from .initialize import get_data_parallel_rank
from .initialize import get_model_parallel_group
from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_world_size


# Default name for the model parallel rng tracker.
_MODEL_PARALLEL_RNG_TRACKER_NAME = 'model-parallel-rng'


# Whether apply model parallelsim to checkpointed hidden states.
_CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER = None
_WORKSPACE_MEMORY_BUFFER = None
_FORWARD_BUFFER = None
_BACKWARD_BUFFER = None
_PARAMETER_GRADIENT_BUFFER = None
_CONJUNCTION_GRADIENT_BUFFER = None
_QKV_FORWARD_BUFFER = None
_QKV_DENSE_BUFFER = None
_H4H_FORWARD_BUFFER = None
_FHH_FORWARD_BUFFER = None
_LMHEAD_DENSE_BUFFER = None


def init_checkpointed_activations_memory_buffer():
    """Initializ the memory buffer for the checkpointed activations."""
    args = get_args()

    per_layer = args.batch_size * args.max_position_embeddings * \
                args.hidden_size // args.model_parallel_size
    assert args.num_layers % args.checkpoint_num_layers == 0, \
        'number of layers is not divisible by checkpoint-num-layers'
    num_checkpointer_layers = args.num_layers // args.checkpoint_num_layers
    numel = per_layer * num_checkpointer_layers
    dtype = torch.half
    if not args.fp16:
        dtype = torch.float

    global _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER
    assert _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER is None, \
        'checkpointed activations memory buffer is already allocated.'
    _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER = allocate_mem_buff(
        'checkpointed activations', numel, dtype, track_usage=False)


def reset_checkpointed_activations_memory_buffer():
    """Reset the memory used for checkpointing."""
    if _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER is not None:
        _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER.reset()


def _set_cuda_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current GPU.

    Argumentss:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)
    else:
        # newer PyTorch
        if device == -1:
            device = torch.device('cuda')
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.cuda.default_generators[idx]
            default_generator.set_state(new_state)

    _lazy_call(cb)


def split_tensor_into_1d_equal_chunks(tensor):
    """Break a tensor into equal 1D chunks."""
    data = tensor.view(-1)
    partition_size = torch.numel(data) // get_model_parallel_world_size()
    start_index = partition_size * get_model_parallel_rank()
    end_index = start_index + partition_size
    return data[start_index:end_index]


def gather_split_1d_tensor(tensor):
    """Opposite of above function, gather values from model parallel ranks."""
    world_size = get_model_parallel_world_size()
    numel = torch.numel(tensor)
    numel_gathered = world_size * numel
    gathered = torch.empty(numel_gathered, dtype=tensor.dtype,
                           device=torch.cuda.current_device(),
                           requires_grad=False)
    chunks = [gathered[i*numel:(i+1)*numel] for i in range(world_size)]
    torch.distributed.all_gather(chunks, tensor,
                                 group=get_model_parallel_group())
    return gathered


class CudaRNGStatesTracker:
    """Tracker for the cuda RNG states.

    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
    """

    def __init__(self):
        # Map from a string name to the cuda rng state.
        self.states_ = {}
        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def reset(self):
        """Set to the initial state (no tracker)."""
        self.states_ = {}
        self.seeds_ = set()

    def get_states(self):
        """Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary."""
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states):
        """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
        self.states_ = states

    def add(self, name, seed):
        """Track the rng state."""
        # Check seed is not already used.
        if seed in self.seeds_:
            raise Exception('seed {} already exists'.format(seed))
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise Exception('cuda rng state {} already exists'.format(name))
        # Get the current rng state.
        orig_rng_state = torch.cuda.get_rng_state()
        # Set the new state and store it.
        torch.cuda.manual_seed(seed)
        self.states_[name] = torch.cuda.get_rng_state()
        # Reset rng state to what it was.
        _set_cuda_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the cuda rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if name not in self.states_:
            raise Exception('cuda rng state {} is not added'.format(name))
        # Store current rng state.
        orig_cuda_rng_state = torch.cuda.get_rng_state()
        # Set rng state to the desired one
        # _set_cuda_rng_state(self.F[name])
        _set_cuda_rng_state(self.states_[name])
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # Update the current rng state for later use.
            self.states_[name] = torch.cuda.get_rng_state()
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state)


# RNG tracker object.
_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()


def get_cuda_rng_tracker():
    """Get cuda rng tracker."""
    return _CUDA_RNG_STATE_TRACKER


def model_parallel_cuda_manual_seed(seed):
    """Initialize model parallel cuda seed.

    This function should be called after the model parallel is
    initialized. Also, no torch.cuda.manual_seed should be called
    after this function. Basically, this is replacement for that
    function.
    Two set of RNG states are tracked:
        default state: This is for data parallelism and is the same among a
                       set of model parallel GPUs but different across
                       different model paralle groups. This is used for
                       example for dropout in the non-model-parallel regions.
        model-parallel state: This state is different among a set of model
                              parallel GPUs, but the same across data parallel
                              groups. This is used for example for dropout in
                              model parallel regions.
    """
    # 2718 is just for fun and any POSITIVE value will work.
    offset = seed + 2718
    model_parallel_seed = offset + get_model_parallel_rank()
    # Data parallel gets the original sedd.
    data_parallel_seed = seed

    if torch.distributed.get_rank() == 0:
        print('> initializing model parallel cuda seeds on global rank {}, '
              'model parallel rank {}, and data parallel rank {} with '
              'model parallel seed: {} and data parallel seed: {}'.format(
                  torch.distributed.get_rank(), get_model_parallel_rank(),
                  get_data_parallel_rank(), model_parallel_seed,
                  data_parallel_seed), flush=True)
    _CUDA_RNG_STATE_TRACKER.reset()
    # Set the default state.
    torch.cuda.manual_seed(data_parallel_seed)
    # and model parallel state.
    _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME,
                                model_parallel_seed)


class CheckpointFunction(torch.autograd.Function):
    """This function is adapted from torch.utils.checkpoint with
       two main changes:
           1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
           2) the states in the model parallel tracker are also properly
              tracked/set/reset.
    """

    @staticmethod
    def forward(ctx, run_function, *args):
        ctx.run_function = run_function

        if _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER is not None:
            ctx.input_0_shape = args[0].data.shape
            args[0].data = _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER.add(
                args[0].data)

        args[0].data = args[0].data.reshape(ctx.input_0_shape)

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()
        ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        if _QKV_FORWARD_BUFFER is not None:
            _QKV_FORWARD_BUFFER.reset()
        if _QKV_DENSE_BUFFER is not None:
            _QKV_DENSE_BUFFER.reset()
        if _H4H_FORWARD_BUFFER is not None:
            _H4H_FORWARD_BUFFER.reset()
        if _FHH_FORWARD_BUFFER is not None:
            _FHH_FORWARD_BUFFER.reset()
        with torch.no_grad():
            outputs = run_function(*args)

        # Store everything.
        ctx.save_for_backward(*args)

        # return outputs_temp
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), "
                               "please use .backward() if possible")
        inputs = ctx.saved_tensors
        # if _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER is not None:
        #     inputs[0].data = inputs[0].data.view(ctx.input_0_shape)

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = torch.cuda.get_rng_state()
        bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        # Compute the forward pass.
        detached_inputs = detach_variable(inputs)
        # if _FORWARD_BUFFER is not None:
        #     _FORWARD_BUFFER.reset()
        if _QKV_FORWARD_BUFFER is not None:
            _QKV_FORWARD_BUFFER.reset()
        if _QKV_DENSE_BUFFER is not None:
            _QKV_DENSE_BUFFER.reset()
        if _H4H_FORWARD_BUFFER is not None:
            _H4H_FORWARD_BUFFER.reset()
        if _FHH_FORWARD_BUFFER is not None:
            _FHH_FORWARD_BUFFER.reset()
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)

        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        if _BACKWARD_BUFFER is not None:
            _BACKWARD_BUFFER.reset()
        torch.autograd.backward(outputs, args)

        # grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp
        #               for inp in detached_inputs)

        if _CONJUNCTION_GRADIENT_BUFFER is not None:
            _CONJUNCTION_GRADIENT_BUFFER.reset()
            grad = _CONJUNCTION_GRADIENT_BUFFER.add(detached_inputs[0].grad)
        grads = (grad, None)

        return (None,) + grads


def checkpoint(function, *args):
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint."""
    return CheckpointFunction.apply(function, *args)


def init_workspace_memory_buffer():
    args = get_args()
    # embedding: [b/q, s], [v/q, h/q] -> [b/q, s, h/q]
    embedding_space = \
        args.padded_vocab_size * args.hidden_size / args.model_parallel_size
        # + args.batch_size * args.seq_length * args.hidden_size / args.model_parallel_size

    # QKV: [b/q, s, h/q], [h/q, 3h/q] -> [b/q, s, 3h/q]
    # QKV_dense: [b/q, s, h/q], [h/q, h/q] -> [b/q, s, b/q]
    # QKV is definitely smaller than mlp

    # mlp: [b/q, s, h/q], [h/q, 4h/q] -> [b/q, s, 4h/q],
    # [b/q, s, 4h/q], [4h/q, h/q] -> [b/q, s, h/q]
    mlp_space = \
        4 * args.batch_size * args.seq_length * args.hidden_size / args.model_parallel_size \
        + 4 * args.hidden_size ** 2 / args.model_parallel_size

    # lm head: [b/q, s, h/q], [v/q, h/q] -> [b/q, s, v/q]
    # space: bsh/p, vh/p, hsv/p
    lm_head_space = \
        args.batch_size * args.seq_length * args.hidden_size / args.model_parallel_size + \
        args.padded_vocab_size * args.hidden_size / args.model_parallel_size

    if args.ParallelTransformer_only:
        workspace = mlp_space
    else:
        workspace = max(embedding_space, mlp_space, lm_head_space)
    workspace = int(workspace)

    name = 'checkpoint workspace'
    dtype = torch.half
    if not args.fp16:
        dtype = torch.float
    global _WORKSPACE_MEMORY_BUFFER
    _WORKSPACE_MEMORY_BUFFER = allocate_mem_buff(
        name, workspace, dtype, track_usage=False)


def get_workspace():
    global _WORKSPACE_MEMORY_BUFFER
    assert _WORKSPACE_MEMORY_BUFFER is not None, \
        'checkpoint workspace is not initialized'
    return _WORKSPACE_MEMORY_BUFFER


def init_forward_buffer():
    args = get_args()
    batch_pp = args.batch_size // args.summa_dim
    seq_length = args.seq_length
    hidden_pp = args.hidden_size // args.summa_dim
    vocab_pp = args.padded_vocab_size // args.summa_dim
    # QKV
    QKV_space = 3 * batch_pp * seq_length * hidden_pp
    # h_to_4h
    h4h_space = 4 * batch_pp * seq_length * hidden_pp
    # 4h_to_h
    fhh_space = batch_pp * seq_length * hidden_pp
    # lm_head
    lm_head_space = batch_pp * seq_length * vocab_pp

    Parallel_transformer_space = QKV_space + h4h_space + fhh_space
    total_space = max(Parallel_transformer_space, lm_head_space)

    name = 'forward buffer'
    dtype = torch.half
    if not args.fp16:
        dtype = torch.float
    global _FORWARD_BUFFER
    _FORWARD_BUFFER = allocate_mem_buff(
        name, total_space, dtype, track_usage=False)


def get_forward_buffer():
    global _FORWARD_BUFFER
    assert _FORWARD_BUFFER is not None, \
        'forward buffer is not initialized'
    return _FORWARD_BUFFER


def init_backward_buffer():
    args = get_args()
    batch_pp = args.batch_size // args.summa_dim
    seq_length = args.seq_length
    hidden_pp = args.hidden_size // args.summa_dim
    vocab_pp = args.padded_vocab_size // args.summa_dim

    # QKV
    QKV_space = batch_pp * seq_length * hidden_pp
    # QKV dense
    QKV_dense_space = batch_pp * seq_length * hidden_pp
    # h_to_4h
    h4h_space = batch_pp * seq_length * hidden_pp
    # 4h_to_h
    fhh_space = 4 * batch_pp * seq_length * hidden_pp
    # lm_head
    lm_head = batch_pp * seq_length * hidden_pp

    Parallel_transformer_space = QKV_space + QKV_dense_space + h4h_space + fhh_space
    if args.ParallelTransformer_only:
        total_space = Parallel_transformer_space
    else:
        total_space = max(Parallel_transformer_space, lm_head)

    name = 'backward buffer'
    dtype = torch.half
    if not args.fp16:
        dtype = torch.float
    global _BACKWARD_BUFFER
    _BACKWARD_BUFFER = allocate_mem_buff(
        name, total_space, dtype, track_usage=False)


def get_backward_buffer():
    global _BACKWARD_BUFFER
    assert _BACKWARD_BUFFER is not None, \
        'backward buffer is not initialized'
    return _BACKWARD_BUFFER


def init_parameter_gradient_buffer():
    args = get_args()
    batch_pp = args.batch_size // args.summa_dim
    seq_length = args.seq_length
    hidden_pp = args.hidden_size // args.summa_dim
    vocab_pp = args.padded_vocab_size // args.summa_dim

    QKV_parameter = 3 * hidden_pp * hidden_pp
    QKV_dense_parameter = hidden_pp * hidden_pp
    h4h_parameter = 4 * hidden_pp * hidden_pp
    fhh_parameter = 4 * hidden_pp * hidden_pp

    Parallel_transformer_parameter = \
        args.num_layers * \
        (QKV_parameter +
         QKV_dense_parameter +
         h4h_parameter +
         fhh_parameter)
    dense_buffer = hidden_pp * hidden_pp
    lm_head_buffer = hidden_pp * vocab_pp
    if args.ParallelTransformer_only:
        total_space = Parallel_transformer_parameter
    else:
        total_space = \
            Parallel_transformer_parameter \
            + lm_head_buffer + dense_buffer

    name = 'parameter gradient buffer'
    dtype = torch.half
    if not args.fp16:
        dtype = torch.float
    global _PARAMETER_GRADIENT_BUFFER
    _PARAMETER_GRADIENT_BUFFER = allocate_mem_buff(
        name, total_space, dtype, track_usage=False)


def get_parameter_gradient_buffer():
    global _PARAMETER_GRADIENT_BUFFER
    assert _PARAMETER_GRADIENT_BUFFER is not None, \
        'parameter gradient is not initialized'
    return _PARAMETER_GRADIENT_BUFFER


def init_conjunction_gradient_buffer():
    args = get_args()
    batch_pp = args.batch_size // args.summa_dim
    seq_length = args.seq_length
    hidden_pp = args.hidden_size // args.summa_dim
    vocab_pp = args.padded_vocab_size // args.summa_dim
    gradient_space = batch_pp * seq_length * hidden_pp

    name = 'conjunction gradient'
    dtype = torch.half
    if not args.fp16:
        dtype = torch.float
    global _CONJUNCTION_GRADIENT_BUFFER
    _CONJUNCTION_GRADIENT_BUFFER = allocate_mem_buff(
        name, gradient_space, dtype, track_usage=False)


def get_conjunction_gradient_buffer():
    global _CONJUNCTION_GRADIENT_BUFFER
    assert _CONJUNCTION_GRADIENT_BUFFER is not None, \
        'conjunctino gradient buffer is not initialized'
    return _CONJUNCTION_GRADIENT_BUFFER


def init_QKV_forward_buffer():
    args = get_args()
    batch_pp = args.batch_size // args.summa_dim
    seq_length = args.seq_length
    hidden_pp = args.hidden_size // args.summa_dim
    global _QKV_FORWARD_BUFFER
    assert _QKV_FORWARD_BUFFER is None, \
        '_QKV_FORWARD_BUFFER is already initialized'
    # shape = (batch_pp, seq_length, 3*hidden_pp)
    space = 3 * batch_pp * seq_length * hidden_pp
    # _QKV_FORWARD_BUFFER = SimpleBuffer(shape, args.params_dtype)
    name = 'QKV forward buffer'
    _QKV_FORWARD_BUFFER = allocate_mem_buff(
        name, space, args.params_dtype, track_usage=False)


def init_QKV_dense_buffer():
    args = get_args()
    batch_pp = args.batch_size // args.summa_dim
    seq_length = args.seq_length
    hidden_pp = args.hidden_size // args.summa_dim
    global _QKV_DENSE_BUFFER
    assert _QKV_DENSE_BUFFER is None, \
        '_QKV_DENSE_BUFFER is already initialized'
    space = batch_pp * seq_length * hidden_pp
    name = 'QKV dense buffer'
    _QKV_DENSE_BUFFER = allocate_mem_buff(
        name, space, args.params_dtype, track_usage=False)


def init_h4h_forward_buffer():
    args = get_args()
    batch_pp = args.batch_size // args.summa_dim
    seq_length = args.seq_length
    hidden_pp = args.hidden_size // args.summa_dim
    vocab_pp = args.padded_vocab_size // args.summa_dim
    global _H4H_FORWARD_BUFFER
    assert _H4H_FORWARD_BUFFER is None, \
        '_H4H_FORWARD_BUFFER is already initialized'
    # shape = (batch_pp, seq_length, 4*hidden_pp)
    space_transformer = 4 * batch_pp * seq_length * hidden_pp
    space_lmhead = batch_pp * seq_length * vocab_pp
    if args.ParallelTransformer_only:
        space = space_transformer
    else:
        space = max(space_transformer, space_lmhead)
    name = 'h4h forward buffer'
    # _H4H_FORWARD_BUFFER = SimpleBuffer(shape, args.params_dtype)
    _H4H_FORWARD_BUFFER = allocate_mem_buff(
        name, space, args.params_dtype, track_usage=False)


def init_fhh_forward_buffer():
    args = get_args()
    batch_pp = args.batch_size // args.summa_dim
    seq_length = args.seq_length
    hidden_pp = args.hidden_size // args.summa_dim
    global _FHH_FORWARD_BUFFER
    assert _FHH_FORWARD_BUFFER is None, \
        '_FHH_FORWARD_BUFFER is already initialized'
    # shape = (batch_pp, seq_length, hidden_pp)
    space = batch_pp * seq_length * hidden_pp
    name = 'fhh forward buffer'
    # _FHH_FORWARD_BUFFER = SimpleBuffer(shape, args.params_dtype)
    _FHH_FORWARD_BUFFER = allocate_mem_buff(
        name, space, args.params_dtype, track_usage=False)


def get_QKV_forward_buffer():
    global _QKV_FORWARD_BUFFER
    assert _QKV_FORWARD_BUFFER is not None, \
        '_QKV_FORWARD_BUFFER is not initialized'
    return _QKV_FORWARD_BUFFER


def get_QKV_dense_buffer():
    global _QKV_DENSE_BUFFER
    assert _QKV_DENSE_BUFFER is not None, \
        '_QKV_DENSE_BUFFER is not initialized'
    return _QKV_DENSE_BUFFER


def get_h4h_forward_buffer():
    global _H4H_FORWARD_BUFFER
    assert _H4H_FORWARD_BUFFER is not None, \
        '_H4H_FORWARD_BUFFER is not initialized'
    return _H4H_FORWARD_BUFFER


def get_fhh_forward_buffer():
    global _FHH_FORWARD_BUFFER
    assert _FHH_FORWARD_BUFFER is not None, \
        '_FHH_FORWARD_BUFFER is not initialized'
    return _FHH_FORWARD_BUFFER


def init_lmhead_dense_buffer():
    args = get_args()
    batch_pp = args.batch_size // args.summa_dim
    seq_length = args.seq_length
    hidden_pp = args.hidden_size // args.summa_dim
    global _LMHEAD_DENSE_BUFFER
    assert _LMHEAD_DENSE_BUFFER is None, \
        '_LMHEAD_DENSE_BUFFER is already initialized'
    space = batch_pp * seq_length * hidden_pp
    name = 'lm-head dense buffer'
    _LMHEAD_DENSE_BUFFER = allocate_mem_buff(
        name, space, args.params_dtype, track_usage=False)


def get_lmhead_dense_buffer():
    global _LMHEAD_DENSE_BUFFER
    assert _LMHEAD_DENSE_BUFFER is not None, \
        '_LMHEAD_DENSE_BUFFER is already initialized'
    return _LMHEAD_DENSE_BUFFER