import torch

from .utils import ensure_divisibility


# Model parallel group that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
# Summa row group that the current rank belongs to
_SUMMA_ROW_GROUP = None
# Summa column group that the current rank belongs to
_SUMMA_COL_GROUP = None
# These values enable us to change the mpu sizes on the fly.
_MPU_WORLD_SIZE = None
_MPU_RANK = None


def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is None


def initialize_model_parallel(model_parallel_size_, summa_dim=2):
    """
    Initialize model data parallel groups.

    Arguments:
        model_parallel_size: number of GPUs used to parallelize model.
        summa_size: GPU mesh size, model_parallel_size = summa_size ** 2

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 4 GPUs to parallelize the model. The present function will
    create 2 model parallel groups and 4 data parallel grous as:
        2 model parallel groups:
            [g0, g1, g2, g3], [g4, g5, g6, g7]
        4 data parallel groups:
            [g0, g4], [g1, g5], [g2, g6], [g3, g7]
        and the model parallel groups are arranged as a square mesh:
        [[g0, g1],    [[g4, g5],
        [g2, g3]]     [g6, g7]]

    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    if torch.distributed.get_rank() == 0:
        print('> initializing model parallel with size {}, summa size {}'.format(
            model_parallel_size_, summa_dim))

    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    model_parallel_size = min(model_parallel_size_, world_size)
    ensure_divisibility(world_size, model_parallel_size)
    assert model_parallel_size == summa_dim ** 2
    rank = torch.distributed.get_rank()

    # Build the data parallel groups.
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, \
        'data parallel group is already initialized'
    for i in range(model_parallel_size):
        ranks = range(i, world_size, model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group

    A = torch.rand((4, 1), dtype=torch.float, device=torch.cuda.current_device())
    with torch.no_grad():
        torch.distributed.broadcast(A,
                                    src=rank % model_parallel_size,
                                    group=_DATA_PARALLEL_GROUP)

    # Build the model parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, \
        'model parallel group is already initialized'
    for i in range(world_size // model_parallel_size):
        ranks = range(i * model_parallel_size,
                      (i + 1) * model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group
    with torch.no_grad():
        torch.distributed.broadcast(A,
                                    src=(rank // model_parallel_size) * model_parallel_size,
                                    group=_MODEL_PARALLEL_GROUP)

    # Build summa row groups.
    global _SUMMA_ROW_GROUP
    assert _SUMMA_ROW_GROUP is None,\
        'summa row group is already initialized'
    for i in range(world_size // model_parallel_size):
        for j in range(summa_dim):
            ranks = range(i * model_parallel_size + j * summa_dim,
                          i * model_parallel_size + (j+1) * summa_dim)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _SUMMA_ROW_GROUP = group

    with torch.no_grad():
        src = ((rank % model_parallel_size) // summa_dim) * summa_dim \
              + (rank // model_parallel_size) * model_parallel_size
        torch.distributed.broadcast(A,
                                    src=src,
                                    group=_SUMMA_ROW_GROUP)

    # Build summa column groups
    global _SUMMA_COL_GROUP
    assert _SUMMA_COL_GROUP is None,\
        'summa column group is already initialized'
    for i in range(world_size // model_parallel_size):
        for j in range(summa_dim):
            ranks = range(i * model_parallel_size + j,
                          (i+1) * model_parallel_size,
                          summa_dim)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _SUMMA_COL_GROUP = group

    with torch.no_grad():
        src = (rank % model_parallel_size) % summa_dim \
              + (rank // model_parallel_size) * model_parallel_size
        torch.distributed.broadcast(A,
                                    src=src,
                                    group=_SUMMA_COL_GROUP)
        torch.distributed.broadcast(A,src=0)


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _MODEL_PARALLEL_GROUP is None or _DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, \
        'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def get_summa_row_group():
    """Get the summa row parallel group the caller rank belongs to."""
    assert _SUMMA_ROW_GROUP is not None, \
        'summa row group is not initiallized'
    return _SUMMA_ROW_GROUP


def get_summa_col_group():
    """Get the summa column parallel group the caller rank belongs to."""
    assert _SUMMA_COL_GROUP is not None, \
        'summa column group is not initialized'
    return _SUMMA_COL_GROUP


def set_model_parallel_world_size(world_size):
    """Set the model parallel size"""
    global _MPU_WORLD_SIZE
    _MPU_WORLD_SIZE = world_size


def get_model_parallel_world_size():
    """Return world size for the model parallel group."""
    global _MPU_WORLD_SIZE
    if _MPU_WORLD_SIZE is not None:
        return _MPU_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_model_parallel_group())


def set_model_parallel_rank(rank):
    """Set model parallel rank."""
    global _MPU_RANK
    _MPU_RANK = rank


def get_model_parallel_rank():
    """Return my rank for the model parallel group."""
    global _MPU_RANK
    if _MPU_RANK is not None:
        return _MPU_RANK
    return torch.distributed.get_rank(group=get_model_parallel_group())


def get_model_parallel_src_rank():
    """Calculate the global rank corresponding to a local rank zeor
    in the model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _SUMMA_ROW_GROUP
    _SUMMA_ROW_GROUP = None
    global _SUMMA_COL_GROUP
    _SUMMA_COL_GROUP = None