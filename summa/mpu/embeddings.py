import torch
import torch.nn.init as init
import summa.mpu as mpu
from summa import get_args
from .utils import VocabUtility
from torch.nn.parameter import Parameter
from .random import get_cuda_rng_tracker
import torch.nn.functional as F
from .layers import _initialize_affine_weight_gpu
from .layers import _initialize_affine_weight_cpu


class VocabParallelEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_,
                 forward_buffer=None, backward_buffer=None,
                 parameter_gradient_buffer=None):
        super(VocabParallelEmbedding, self).__init__()
        self.vocab_size = num_embeddings
        self.hidden_size = embedding_dim
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None

        args = get_args()
        self.summa_dim = args.summa_dim
        self.model_parallel_size = args.model_parallel_size
        self.col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
        self.row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
        self.ddp_rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())

        if forward_buffer is None:
            self.forward_buffer = mpu.get_QKV_dense_buffer()
        else:
            self.forward_buffer = forward_buffer

        if backward_buffer is None:
            self.backward_buffer = mpu.get_backward_buffer()
        else:
            self.backward_buffer = backward_buffer

        if parameter_gradient_buffer is None:
            self.parameter_gradient_buffer = mpu.get_parameter_gradient_buffer()
        else:
            self.parameter_gradient_buffer = parameter_gradient_buffer

        # Settings for weight
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.vocab_size, self.row_rank, self.summa_dim)

        self.num_embeddings_per_partition = \
            self.vocab_end_index - self.vocab_start_index

        self.hidden_size_per_partition = mpu.divide(
            self.hidden_size, self.summa_dim)

        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition,
                self.hidden_size_per_partition,
                device=torch.cuda.current_device(),
                dtype=args.params_dtype))
            self.weight_master = _initialize_affine_weight_cpu(
                self.weight, self.vocab_size, self.hidden_size,
                init_method, self.row_rank, self.col_rank)
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition,
                self.hidden_size_per_partition,
                device=torch.cuda.current_device(),
                dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method, stride=1)

    def forward(self, idx):
        # idx: [b, s]
        batch_size = idx.size(0)
        batch_size_per_partition = mpu.divide(batch_size, self.summa_dim)
        batch_start = batch_size_per_partition * self.row_rank
        batch_end = batch_start + batch_size_per_partition
        # idx: [b/q, s]
        idx = idx[batch_start: batch_end, :]
        output = Embedding.apply(idx, self.weight,
                                 self.row_rank, self.col_rank, self.ddp_rank,
                                 self.summa_dim, self.model_parallel_size,
                                 batch_size_per_partition, idx.size(1),
                                 self.num_embeddings_per_partition,
                                 self.hidden_size_per_partition,
                                 self.forward_buffer, None, self.parameter_gradient_buffer)
        return output


class Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, idx, weight, row_rank, col_rank, ddp_rank, summa_dim, model_parallel_size,
                batch_pp, seq_length, vocab_pp, hidden_pp, forward_buffer=None, backward_buffer=None,
                parameter_gradient_buffer=None):
        # idx: [b/q, s], weight: [v/q, h/q]
        idx = idx.view(-1) # [bs/q]
        arange = torch.arange(start=0, end=batch_pp*seq_length,
                              dtype=torch.int64, device=weight.device).unsqueeze(0)
        out_size = (batch_pp*seq_length, hidden_pp)
        output = forward_buffer.zeros(out_size)
        workspace = mpu.get_workspace()
        with torch.no_grad():
            for i in range(summa_dim):
                workspace.reset()
                idx_temp = idx.clone()
                vocab_start = vocab_pp * i
                vocab_end = vocab_start + vocab_pp
                mask = (idx_temp < vocab_start) | (idx_temp >= vocab_end)
                idx_temp -= vocab_start
                idx_temp[mask] = 0
                idx_temp = idx_temp.unsqueeze(0)
                indices = torch.cat((arange, idx_temp), dim=0)
                value = torch.ones(batch_pp*seq_length, dtype=weight.dtype)
                value[mask] = 0
                idx_sparse = torch.sparse_coo_tensor(indices, value, (batch_pp*seq_length, vocab_pp))
                idx_sparse = idx_sparse.to(device=torch.cuda.current_device())

                weight_temp = workspace.add(weight)
                torch.distributed.broadcast(
                    weight_temp,
                    src=i*summa_dim+col_rank+ddp_rank*model_parallel_size,
                    group=mpu.get_summa_col_group())
                torch.addmm(output, idx_sparse, weight_temp, out=output)
        output_shape = (batch_pp, seq_length, hidden_pp)
        output = output.reshape(output_shape)
        ctx.save_for_backward(idx, weight, arange)
        ctx.row_rank = row_rank
        ctx.col_rank = col_rank
        ctx.ddp_rank = ddp_rank
        ctx.summa_dim = summa_dim
        ctx.model_parallel_size = model_parallel_size
        ctx.batch_pp = batch_pp
        ctx.seq_length = seq_length
        ctx.vocab_pp = vocab_pp
        ctx.hidden_pp = hidden_pp
        ctx.backward_buffer = backward_buffer
        ctx.parameter_gradient_buffer = parameter_gradient_buffer
        return output

    @staticmethod
    def backward(ctx, output_grad):
        idx, weight, arange = ctx.saved_tensors # [bs/q], [v/q, h/q], [bs/q]
        input_size = (ctx.vocab_pp, ctx.hidden_pp)
        input_grad = ctx.parameter_gradient_buffer.zeros(input_size)
        output_grad_shape = (ctx.batch_pp*ctx.seq_length, ctx.hidden_pp)
        output_grad = output_grad.reshape(output_grad_shape)
        workspace = mpu.get_workspace()
        with torch.no_grad():
            for i in range(ctx.summa_dim):
                workspace.reset()
                idx_temp = idx.clone()
                out_temp = workspace.zeros(input_size)
                vocab_start = ctx.vocab_pp * i
                vocab_end = vocab_start + ctx.vocab_pp
                mask = (idx_temp < vocab_start) | (idx_temp >= vocab_end)
                idx_temp -= vocab_start
                idx_temp[mask] = 0
                idx_temp = idx_temp.unsqueeze(0)
                indices = torch.cat((idx_temp, arange), dim=0)
                value = torch.ones(ctx.batch_pp * ctx.seq_length, dtype=weight.dtype)
                value[mask] = 0
                idx_sparse = torch.sparse_coo_tensor(
                    indices, value, (ctx.vocab_pp, ctx.batch_pp*ctx.seq_length))
                idx_sparse = idx_sparse.to(device=torch.cuda.current_device())

                torch.matmul(idx_sparse, output_grad, out=out_temp)
                torch.distributed.reduce(
                    out_temp,
                    dst=ctx.col_rank+i*ctx.summa_dim+ctx.ddp_rank*ctx.model_parallel_size,
                    group=mpu.get_summa_col_group())
                if ctx.row_rank == i:
                    input_grad.copy_(out_temp)
        return None, input_grad, None, None, None, None, None, None, None, None, None, None, None, None


class Vocab_Position_Tokentype_ParallelEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_tokentypes,
                 init_method=init.xavier_normal_,
                 forward_buffer=None, backward_buffer=None,
                 parameter_gradient_buffer=None):
        super(Vocab_Position_Tokentype_ParallelEmbedding, self).__init__()
        self.vocab_size = num_embeddings
        self.hidden_size = embedding_dim
        self.num_tokentypes = num_tokentypes
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None

        args = get_args()
        self.summa_dim = args.summa_dim
        self.model_parallel_size = args.model_parallel_size
        self.col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
        self.row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
        self.ddp_rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())

        if forward_buffer is None:
            self.forward_buffer = mpu.get_QKV_dense_buffer()
            # self.forward_buffer = mpu.get_checkpoint_activation_buffer()
        else:
            self.forward_buffer = forward_buffer

        if backward_buffer is None:
            self.backward_buffer = mpu.get_backward_buffer()
        else:
            self.backward_buffer = backward_buffer

        if parameter_gradient_buffer is None:
            self.parameter_gradient_buffer = mpu.get_parameter_gradient_buffer()
        else:
            self.parameter_gradient_buffer = parameter_gradient_buffer

        # Settings for weight
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.vocab_size, self.row_rank, self.summa_dim)

        self.num_embeddings_per_partition = \
            self.vocab_end_index - self.vocab_start_index

        self.hidden_size_per_partition = mpu.divide(
            self.hidden_size, self.summa_dim)

        if args.use_cpu_initialization:
            self.vocab_weight = Parameter(torch.empty(
                self.num_embeddings_per_partition,
                self.hidden_size_per_partition,
                device=torch.cuda.current_device(),
                dtype=args.params_dtype))
            self.vocab_weight_master = _initialize_affine_weight_cpu(
                self.vocab_weight, self.vocab_size, self.hidden_size,
                init_method, self.row_rank, self.col_rank)
        else:
            self.vocab_weight = Parameter(torch.empty(
                self.num_embeddings_per_partition,
                self.hidden_size_per_partition,
                device=torch.cuda.current_device(),
                dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.vocab_weight, init_method, stride=1)

        # positional embedding weight initialization
        if self.row_rank == 0:
            self.pos_weight = Parameter(torch.empty(
                args.seq_length,
                self.hidden_size_per_partition,
                device=torch.cuda.current_device(),
                dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.pos_weight, init_method, stride=1)
        else:
            self.pos_weight = None

        # tokentype weight initialization
        if self.row_rank == 0:
            self.tokentype_weight = Parameter(torch.empty(
                self.num_tokentypes,
                self.hidden_size_per_partition,
                device=torch.cuda.current_device(),
                dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.tokentype_weight, init_method, stride=1)
        else:
            self.tokentype_weight = None

    def forward(self, idx, types):
        # idx: [b, s]
        batch_size = idx.size(0)
        batch_size_per_partition = mpu.divide(batch_size, self.summa_dim)
        batch_start = batch_size_per_partition * self.row_rank
        batch_end = batch_start + batch_size_per_partition
        # idx: [b/q, s]
        idx = idx[batch_start: batch_end, :]
        types = types[batch_start:batch_end, :]
        output = PPTEmbedding.apply(idx, types, self.vocab_weight, self.pos_weight, self.tokentype_weight,
                                 self.row_rank, self.col_rank, self.ddp_rank,
                                 self.summa_dim, self.model_parallel_size,
                                 batch_size_per_partition, idx.size(1),
                                 self.num_embeddings_per_partition,
                                 self.hidden_size_per_partition, self.num_tokentypes,
                                 self.forward_buffer, None, self.parameter_gradient_buffer)
        return output


class PPTEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, idx, types, vocab_weight, pos_weight, tokentype_weight,
                row_rank, col_rank, ddp_rank, summa_dim, model_parallel_size,
                batch_pp, seq_length, vocab_pp, hidden_pp, num_tokentypes,
                forward_buffer=None, backward_buffer=None,
                parameter_gradient_buffer=None):
        # idx: [b/q, s], weight: [v/q, h/q]
        idx = idx.view(-1) # [bs/q]
        arange = torch.arange(start=0, end=batch_pp*seq_length,
                              dtype=torch.int64, device=vocab_weight.device).unsqueeze(0)
        out_size = (batch_pp*seq_length, hidden_pp)
        # output = forward_buffer.zeros(out_size)
        output = torch.zeros(out_size, dtype=vocab_weight.dtype, device=vocab_weight.device)
        workspace = mpu.get_workspace()
        # calculate tokentype embedding
        with torch.no_grad():
            for i in range(summa_dim):
                workspace.reset()
                idx_temp = idx.clone()
                vocab_start = vocab_pp * i
                vocab_end = vocab_start + vocab_pp
                mask = (idx_temp < vocab_start) | (idx_temp >= vocab_end)
                idx_temp -= vocab_start
                idx_temp[mask] = 0
                idx_temp = idx_temp.unsqueeze(0)
                indices = torch.cat((arange, idx_temp), dim=0)
                value = torch.ones(batch_pp*seq_length, dtype=vocab_weight.dtype)
                value[mask] = 0
                idx_sparse = torch.sparse_coo_tensor(indices, value, (batch_pp*seq_length, vocab_pp))
                idx_sparse = idx_sparse.to(device=torch.cuda.current_device())

                weight_temp = workspace.add(vocab_weight)
                torch.distributed.broadcast(
                    weight_temp,
                    src=i*summa_dim+col_rank+ddp_rank*model_parallel_size,
                    group=mpu.get_summa_col_group())
                torch.addmm(output, idx_sparse, weight_temp, out=output)
        output_shape = (batch_pp, seq_length, hidden_pp)
        output = output.reshape(output_shape)

        # calculate positional embedding
        with torch.no_grad():
            workspace.reset()
            if pos_weight is None:
                weight_temp = workspace.zeros((seq_length, hidden_pp))
            else:
                weight_temp = workspace.add(pos_weight)
            torch.distributed.broadcast(
                weight_temp, src=col_rank+ddp_rank*model_parallel_size,
                group=mpu.get_summa_col_group())
            output += weight_temp.unsqueeze(0)

        # calculate token type embedding
        with torch.no_grad():
            types = types.view(-1).unsqueeze(0) # [1, bs/q]
            indices = torch.cat((arange, types), dim=0)
            value = torch.ones(batch_pp*seq_length)
            types_sparse = torch.sparse_coo_tensor(indices, value, (batch_pp*seq_length, num_tokentypes))
            types_sparse = types_sparse.to(device=torch.cuda.current_device())

            workspace.reset()
            if tokentype_weight is None:
                weight_temp = workspace.zeros((num_tokentypes, hidden_pp))
            else:
                weight_temp = workspace.add(tokentype_weight)
            torch.distributed.broadcast(
                weight_temp, src=col_rank+ddp_rank*model_parallel_size,
                group=mpu.get_summa_col_group())
            output = output.reshape((batch_pp * seq_length, hidden_pp))
            torch.addmm(output, types_sparse, weight_temp, out=output)
            output = output.reshape((batch_pp, seq_length, hidden_pp))

        ctx.save_for_backward(idx, types, vocab_weight, pos_weight, tokentype_weight, arange)
        ctx.row_rank = row_rank
        ctx.col_rank = col_rank
        ctx.ddp_rank = ddp_rank
        ctx.summa_dim = summa_dim
        ctx.model_parallel_size = model_parallel_size
        ctx.batch_pp = batch_pp
        ctx.seq_length = seq_length
        ctx.vocab_pp = vocab_pp
        ctx.hidden_pp = hidden_pp
        ctx.num_tokentypes = num_tokentypes
        ctx.backward_buffer = backward_buffer
        ctx.parameter_gradient_buffer = parameter_gradient_buffer
        return output

    @staticmethod
    def backward(ctx, output_grad):
        idx, types, vocab_weight, pos_weight, tokentype_weight, arange = ctx.saved_tensors # [1, bs/q], [v/q, h/q], [bs/q]
        input_size = (ctx.vocab_pp, ctx.hidden_pp)
        input_grad = ctx.parameter_gradient_buffer.zeros(input_size)
        output_grad_shape = (ctx.batch_pp*ctx.seq_length, ctx.hidden_pp)
        output_grad = output_grad.reshape(output_grad_shape)
        workspace = mpu.get_workspace()
        with torch.no_grad():
            for i in range(ctx.summa_dim):
                workspace.reset()
                idx_temp = idx.clone()
                out_temp = workspace.zeros(input_size)
                vocab_start = ctx.vocab_pp * i
                vocab_end = vocab_start + ctx.vocab_pp
                mask = (idx_temp < vocab_start) | (idx_temp >= vocab_end)
                idx_temp -= vocab_start
                idx_temp[mask] = 0
                idx_temp = idx_temp.unsqueeze(0)
                indices = torch.cat((idx_temp, arange), dim=0)
                value = torch.ones(ctx.batch_pp * ctx.seq_length, dtype=vocab_weight.dtype)
                value[mask] = 0
                idx_sparse = torch.sparse_coo_tensor(
                    indices, value, (ctx.vocab_pp, ctx.batch_pp*ctx.seq_length))
                idx_sparse = idx_sparse.to(device=torch.cuda.current_device())

                torch.matmul(idx_sparse, output_grad, out=out_temp)
                torch.distributed.reduce(
                    out_temp,
                    dst=ctx.col_rank+i*ctx.summa_dim+ctx.ddp_rank*ctx.model_parallel_size,
                    group=mpu.get_summa_col_group())
                if ctx.row_rank == i:
                    input_grad.copy_(out_temp)

        # calculate positional embedding backward
        with torch.no_grad():
            output_grad = output_grad.reshape((ctx.batch_pp, ctx.seq_length, ctx.hidden_pp))
            pos_weight_grad = torch.sum(output_grad, dim=0)
            torch.distributed.reduce(
                pos_weight_grad,
                dst=ctx.col_rank+ctx.ddp_rank*ctx.model_parallel_size,
                group=mpu.get_summa_col_group())
            if ctx.row_rank != 0:
                pos_weight_grad = None

        # calculate tokentype embedding backard
        with torch.no_grad():
            tokentype_weight_grad = ctx.parameter_gradient_buffer.zeros((ctx.num_tokentypes, ctx.hidden_pp))
            indices = torch.cat((types, arange), dim=0)
            value = torch.ones(ctx.batch_pp * ctx.seq_length)
            types_sparse = torch.sparse_coo_tensor(indices, value, (ctx.num_tokentypes, ctx.batch_pp*ctx.seq_length))
            types_sparse = types_sparse.to(device=torch.cuda.current_device())
            output_grad = output_grad.reshape((ctx.batch_pp*ctx.seq_length, ctx.hidden_pp))
            torch.addmm(tokentype_weight_grad, types_sparse, output_grad, out=tokentype_weight_grad)
            torch.distributed.reduce(
                tokentype_weight_grad,
                dst=ctx.col_rank+ctx.ddp_rank*ctx.model_parallel_size,
                group=mpu.get_summa_col_group())
            if ctx.row_rank != 0:
                tokentype_weight_grad = None

        return None, None, input_grad, pos_weight_grad, tokentype_weight_grad, None, None, None, None, None, None, None, None, None, None, None, None, None