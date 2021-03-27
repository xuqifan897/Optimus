import torch
import torch.nn.init as init
import summa.mpu as mpu
from summa import get_args
from .utils import VocabUtility
from torch.nn.parameter import Parameter
from .random import get_cuda_rng_tracker
import torch.nn.functional as F

def _initialize_affine_weight_gpu(weight, init_method, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    weight.model_parallel = True
    weight.partition_stride = stride

    with get_cuda_rng_tracker().fork():
        init_method(weight)


def _initialize_affine_weight_cpu(weight, output_size, input_size,
                                  init_method, row_rank, col_rank, stride=1):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    weight.model_parallel = True
    weight.partition_stride = stride

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=torch.float,
                                requires_grad=False,
                                device=torch.cuda.current_device())
    init_method(master_weight)
    args = get_args()
    master_weight = master_weight.to(dtype=args.params_dtype)

    # Split and copy
    output_size_per_partition = mpu.divide(output_size, args.summa_dim)
    input_size_per_partition = mpu.divide(input_size, args.summa_dim)
    weight_list_1 = torch.split(master_weight, output_size_per_partition,
                                dim=0)
    weight_1 = weight_list_1[row_rank]
    weight_2_list = torch.split(weight_1, input_size_per_partition, dim=1)
    weight_2 = [weight_2_list[col_rank]]
    with torch.no_grad():
        torch.cat(weight_2, dim=0, out=weight)

    return master_weight


class VocabParallelEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_):
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
                                 self.hidden_size_per_partition)
        return output


class Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, idx, weight,
                row_rank, col_rank, ddp_rank, summa_dim, model_parallel_size,
                batch_pp, seq_length, vocab_pp, hidden_pp):
        # idx: [b/q, s], weight: [v/q, h/q]
        out_size = (batch_pp, seq_length, hidden_pp)
        output = torch.zeros(out_size,
                             dtype=weight.dtype,
                             device=torch.cuda.current_device())
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

                weight_temp = workspace.add(weight)
                torch.distributed.broadcast(
                    weight_temp,
                    src=i*summa_dim+col_rank+ddp_rank*model_parallel_size,
                    group=mpu.get_summa_col_group())
                output_temp = F.embedding(idx_temp, weight_temp)
                output_temp[mask, :] = 0.0
                output.add_(output_temp)

        ctx.save_for_backward(idx, weight)
        ctx.row_rank = row_rank
        ctx.col_rank = col_rank
        ctx.ddp_rank = ddp_rank
        ctx.model_parallel_size = model_parallel_size
        ctx.summa_dim = summa_dim
        ctx.batch_pp = batch_pp
        ctx.seq_length = seq_length
        ctx.vocab_pp = vocab_pp
        ctx.hidden_pp = hidden_pp

        return output

    @staticmethod
    def backward(ctx, output_grad):
        idx, weight = ctx.saved_tensors # idx: [b/q, s], weight: [v/q, h/q]
        out_size = (ctx.batch_pp, ctx.seq_length, ctx.hidden_pp)
        workspace = mpu.get_workspace()
        for i in range(ctx.summa_dim):
            with torch.no_grad():
                workspace.reset()
                idx_temp = idx.clone()
                vocab_start = ctx.vocab_pp * i
                vocab_end = vocab_start + ctx.vocab_pp
                mask = (idx_temp < vocab_start) | (idx_temp >= vocab_end)
                idx_temp -= vocab_start
                idx_temp[mask] = 0

                weight_temp = workspace.add(weight)
                torch.distributed.broadcast(
                    weight_temp,
                    src=i*ctx.summa_dim+ctx.col_rank+ctx.ddp_rank*ctx.model_parallel_size,
                    group=mpu.get_summa_col_group())
            with torch.enable_grad():
                weight_temp.requires_grad = True
                out_temp = F.embedding(idx_temp, weight_temp)
                out_temp[mask, :] = 0.0
                out_temp.backward(output_grad)
            with torch.no_grad():
                torch.distributed.reduce(
                    weight_temp.grad,
                    dst=i*ctx.summa_dim+ctx.col_rank+ctx.ddp_rank*ctx.model_parallel_size,
                    group=mpu.get_summa_col_group())
                if ctx.row_rank == i:
                    weight_grad = weight_temp.grad
        return None, weight_grad, None, None, None, None, None, None, None, None, None


class PosParallelEmbedding(torch.nn.Module):
    def __init__(self, seq_length, hidden_size, init_method):
        super(PosParallelEmbedding, self).__init__()
        args=get_args()
        self.summa_dim = args.summa_dim
        self.model_parallel_size = args.model_parallel_size
        self.col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
        self.row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
        self.ddp_rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
        self.seq_length = seq_length
        self.hidden_pp = mpu.divide(hidden_size, self.summa_dim)
        if self.row_rank == 0:
            self.weight = Parameter(torch.empty(
                (seq_length, self.hidden_pp),
                device=torch.cuda.current_device(),
                dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method, stride=1)
        else:
            self.weight = None

    def forward(self, hidden):
        # hidden: [b/q, s, h/q]
        output = PPE.apply(hidden, self.weight,
                           self.row_rank, self.col_rank, self.ddp_rank,
                           self.summa_dim, self.model_parallel_size,
                           self.seq_length, self.hidden_pp)
        return output


class PPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight,
                row_rank, col_rank, ddp_rank, summa_dim, model_parallel_size,
                seq_length, hidden_pp):
        # input: [b/q, s, h/q]
        # weight: [s, h/q]
        workspace = mpu.get_workspace()
        workspace.reset()
        if row_rank == 0:
            weight_temp = workspace.add(weight)
        else:
            weight_temp = workspace.zeros((seq_length, hidden_pp))
        torch.distributed.broadcast(
            weight_temp,
            src=col_rank+ddp_rank*model_parallel_size,
            group=mpu.get_summa_col_group())

        ctx.row_rank = row_rank
        ctx.col_rank = col_rank
        ctx.ddp_rank = ddp_rank
        ctx.summa_dim = summa_dim
        ctx.model_parallel_size = model_parallel_size
        ctx.seq_length = seq_length
        ctx.hidden_pp = hidden_pp

        return input + weight_temp

    @staticmethod
    def backward(ctx, output_grad):
        # output_grad: [b/q, s, h/q]
        workspace = mpu.get_workspace()
        workspace.reset()
        grad_temp = workspace.zeros((ctx.seq_length, ctx.hidden_pp))
        torch.sum(output_grad, dim=0, out=grad_temp)
        torch.distributed.reduce(
            grad_temp,
            dst=ctx.col_rank+ctx.ddp_rank*ctx.model_parallel_size,
            group=mpu.get_summa_col_group())
        if ctx.row_rank == 0:
            return output_grad, grad_temp, None, None, None, None, None, None, None
        else:
            return output_grad, None, None, None, None, None, None, None, None,


class TokentypeParallelEmbedding(torch.nn.Module):
    def __init__(self, num_tokentypes, hidden_size, init_method):
        super(TokentypeParallelEmbedding, self).__init__()
        args = get_args()
        self.col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
        self.row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
        self.ddp_rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
        self.summa_dim = args.summa_dim
        self.model_parallel_size = args.model_parallel_size
        self.hidden_pp = mpu.divide(hidden_size, self.summa_dim)
        self.num_tokentypes = num_tokentypes

        if self.row_rank == 0:
            self.weight = Parameter(torch.empty(
                (num_tokentypes, self.hidden_pp),
                device=torch.cuda.current_device(),
                dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method, stride=1)
        else:
            self.weight = Parameter(torch.tensor(
                0,
                device=torch.cuda.current_device(),
                dtype=args.params_dtype))

    def forward(self, tokentype_ids):
        # idx: [b, s]
        batch_size = tokentype_ids.size(0)
        batch_size_per_partition = mpu.divide(batch_size, self.summa_dim)
        batch_start = batch_size_per_partition * self.row_rank
        batch_end = batch_start + batch_size_per_partition
        # idx: [b/q, s]
        tokentype_ids = tokentype_ids[batch_start: batch_end, :]
        out = TPE.apply(tokentype_ids, self.weight,
                        self.row_rank, self.col_rank, self.ddp_rank,
                        self.summa_dim, self.model_parallel_size,
                        self.num_tokentypes, self.hidden_pp)
        return out


class TPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokentype_ids, weight,
                row_rank, col_rank, ddp_rank, summa_dim, model_parallel_size,
                num_tokentypes, hidden_pp):
        workspace = mpu.get_workspace()
        workspace.reset()
        if row_rank == 0:
            weight_temp = workspace.add(weight)
        else:
            weight_temp = workspace.zeros((num_tokentypes, hidden_pp))
        with torch.no_grad():
            torch.distributed.broadcast(
                weight_temp,
                src=col_rank+ddp_rank*model_parallel_size,
                group=mpu.get_summa_col_group())
        out = F.embedding(tokentype_ids, weight_temp)
        ctx.save_for_backward(tokentype_ids, weight)
        ctx.row_rank = row_rank
        ctx.col_rank = col_rank
        ctx.ddp_rank = ddp_rank
        ctx.summa_dim = summa_dim
        ctx.model_parallel_size = model_parallel_size
        ctx.num_tokentypes = num_tokentypes
        ctx.hidden_pp = hidden_pp
        return out

    @staticmethod
    def backward(ctx, output_grad):
        tokentype_ids, weight = ctx.saved_tensors
        workspace = mpu.get_workspace()
        workspace.reset()
        if ctx.row_rank == 0:
            weight_temp = workspace.add(weight)
        else:
            weight_temp = workspace.zeros((ctx.num_tokentypes, ctx.hidden_pp))
        with torch.no_grad():
            torch.distributed.broadcast(
                weight_temp,
                src=ctx.col_rank+ctx.ddp_rank*ctx.model_parallel_size,
                group=mpu.get_summa_col_group())
        with torch.enable_grad():
            weight_temp.requires_grad = True
            out = F.embedding(tokentype_ids, weight_temp)
            out.backward(output_grad)
        with torch.no_grad():
            torch.distributed.reduce(
                weight_temp.grad,
                dst=ctx.col_rank+ctx.ddp_rank*ctx.model_parallel_size,
                group=mpu.get_summa_col_group())
        if ctx.row_rank == 0:
            return None, weight_temp.grad, None, None, None, None, None, None, None
        else:
            return None, None, None, None, None, None, None, None, None


class SUMMA_AB(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, C_shape, row_rank, col_rank,
                ddp_rank, summa_dim, model_parallel_size,
                forward_buffer, backward_buffer,
                parameter_gradient_buffer):
        torch.cuda.empty_cache()
        ctx.save_for_backward(A, B)
        if (A.dtype is torch.float32) or (B.dtype is torch.float32):
            dtype = torch.float32
        else:
            dtype = torch.float16
        A_shape = A.shape
        A = A.reshape((-1, A_shape[-1]))
        B_shape = B.shape
        B = B.reshape((-1, B_shape[-1]))
        out_shape = (A.shape[0], B.shape[1])
        if forward_buffer is None:
            out = torch.zeros(
                out_shape, dtype=dtype,
                device=torch.cuda.current_device())
        else:
            out = forward_buffer.zeros(out_shape)
        workspace = mpu.get_workspace()
        for i in range(summa_dim):
            workspace.reset()
            A_temp = workspace.add(A)
            B_temp = workspace.add(B)
            # pre-allocate output memory
            torch.distributed.broadcast(A_temp,
                                        src=i+row_rank*summa_dim+ddp_rank*model_parallel_size,
                                        group=mpu.get_summa_row_group())
            torch.distributed.broadcast(B_temp,
                                        src=col_rank+i*summa_dim+ddp_rank*model_parallel_size,
                                        group=mpu.get_summa_col_group())
            torch.addmm(out, A_temp, B_temp, out=out)
        out = out.reshape(C_shape)
        ctx.row_rank = row_rank
        ctx.col_rank = col_rank
        ctx.ddp_rank = ddp_rank
        ctx.summa_dim = summa_dim
        ctx.model_parallel_size = model_parallel_size
        ctx.backward_buffer = backward_buffer
        ctx.parameter_gradient_buffer = parameter_gradient_buffer
        return out

    @staticmethod
    def backward(ctx, output_grad):
        A, B = ctx.saved_tensors
        with torch.no_grad():
            A_grad = SUMMA_ABT.apply(
                output_grad, B, A.shape, ctx.row_rank, ctx.col_rank,
                ctx.ddp_rank, ctx.summa_dim, ctx.model_parallel_size,
                ctx.backward_buffer, None, None)
            B_grad = SUMMA_ATB.apply(
                A, output_grad, B.shape, ctx.row_rank, ctx.col_rank,
                ctx.ddp_rank, ctx.summa_dim, ctx.model_parallel_size,
                ctx.parameter_gradient_buffer, None, None)
        return A_grad, B_grad, None, None, None, None, None, None, None, None, None


class SUMMA_ABT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, C_shape, row_rank, col_rank,
                ddp_rank, summa_dim, model_parallel_size,
                forward_buffer, backward_buffer,
                parameter_gradient_buffer):
        torch.cuda.empty_cache()
        ctx.save_for_backward(A, B)
        if (A.dtype is torch.float32) or (B.dtype is torch.float32):
            dtype = torch.float32
        else:
            dtype = torch.float16
        A_shape = A.shape
        A = A.reshape(-1, A_shape[-1])
        B_shape = B.shape
        B = B.reshape(-1, B_shape[-1])
        out_shape = (A.shape[0], B.shape[0])
        if forward_buffer is None:
            out = torch.zeros(
                out_shape, dtype=dtype,
                device=torch.cuda.current_device())
        else:
            out = forward_buffer.zeros(out_shape)
        workspace = mpu.get_workspace()
        for i in range(summa_dim):
            workspace.reset()
            B_temp = workspace.add(B)
            C_temp = workspace.zeros(out_shape)
            torch.distributed.broadcast(B_temp,
                                        src=col_rank+i*summa_dim+ddp_rank*model_parallel_size,
                                        group=mpu.get_summa_col_group())
            torch.matmul(A, B_temp.transpose(0, 1), out=C_temp)
            torch.distributed.reduce(C_temp,
                                     dst=i+row_rank*summa_dim+ddp_rank*model_parallel_size,
                                     group=mpu.get_summa_row_group())
            if col_rank == i:
                out.copy_(C_temp)
        out = out.reshape(C_shape)
        ctx.row_rank = row_rank
        ctx.col_rank = col_rank
        ctx.ddp_rank = ddp_rank
        ctx.summa_dim = summa_dim
        ctx.model_parallel_size = model_parallel_size
        ctx.forward_buffer = forward_buffer
        ctx.backward_buffer = backward_buffer
        ctx.parameter_gradient_buffer = parameter_gradient_buffer
        return out

    @staticmethod
    def backward(ctx, output_grad):
        A, B = ctx.saved_tensors
        with torch.no_grad():
            A_grad = SUMMA_AB.apply(
                output_grad, B, A.shape, ctx.row_rank, ctx.col_rank,
                ctx.ddp_rank, ctx.summa_dim, ctx.model_parallel_size,
                ctx.backward_buffer, None, None)
            B_grad = SUMMA_ATB.apply(
                output_grad, A, B.shape, ctx.row_rank, ctx.col_rank,
                ctx.ddp_rank, ctx.summa_dim, ctx.model_parallel_size,
                ctx.parameter_gradient_buffer, None, None)
        return A_grad, B_grad, None, None, None, None, None, None, None, None, None


class SUMMA_ATB(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, C_shape, row_rank, col_rank,
                ddp_rank, summa_dim, model_parallel_size,
                forward_buffer, backward_buffer,
                parameter_gradient_buffer):
        torch.cuda.empty_cache()
        ctx.save_for_backward(A, B)
        if (A.dtype is torch.float32) or (B.dtype is torch.float32):
            dtype = torch.float32
        else:
            dtype = torch.float16
        A_shape = A.shape
        A = A.reshape(-1, A_shape[-1])
        B_shape = B.shape
        B = B.reshape(-1, B_shape[-1])
        out_shape = (A.shape[1], B.shape[1])
        if forward_buffer is None:
            out = torch.zeros(
                out_shape, dtype=dtype,
                device=torch.cuda.current_device())
        else:
            out = forward_buffer.zeros(out_shape)
        workspace = mpu.get_workspace()
        for i in range(summa_dim):
            workspace.reset()
            A_temp = workspace.add(A)
            C_temp = workspace.zeros(out_shape)
            torch.distributed.broadcast(A_temp,
                                        src=i+row_rank*summa_dim+ddp_rank*model_parallel_size,
                                        group=mpu.get_summa_row_group())
            torch.matmul(A_temp.transpose(0, 1), B, out=C_temp)
            torch.distributed.reduce(C_temp,
                                     dst=col_rank+i*summa_dim+ddp_rank*model_parallel_size,
                                     group=mpu.get_summa_col_group())
            if i == row_rank:
                out.copy_(C_temp)
        out = out.reshape(C_shape)
        ctx.row_rank = row_rank
        ctx.col_rank = col_rank
        ctx.ddp_rank = ddp_rank
        ctx.summa_dim = summa_dim
        ctx.model_parallel_size = model_parallel_size
        return out

    @staticmethod
    def backward(ctx, output_grad):
        A, B = ctx.saved_tensors
        with torch.no_grad():
            A_grad = SUMMA_ABT.apply(
                B, output_grad, A.shape, ctx.row_rank, ctx.col_rank,
                ctx.ddp_rank, ctx.summa_dim, ctx.model_parallel_size,
                None, None, None)
            B_grad = SUMMA_AB.apply(
                A, output_grad, B.shape, ctx.row_rank, ctx.col_rank,
                ctx.ddp_rank, ctx.summa_dim, ctx.model_parallel_size,
                None, None, None)
        return A_grad, B_grad, None, None, None, None, None, None, None, None, None


class SUMMAbias(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias, output_size_per_partition,
                row_rank, col_rank, ddp_rank, model_parallel_size,
                skip_bias_add, dtype):
        with torch.no_grad():
            if row_rank == 0:
                bias_temp = bias.clone()
            else:
                bias_temp = torch.zeros(
                    output_size_per_partition,
                    dtype=dtype,
                    device=torch.cuda.current_device())
            torch.distributed.broadcast(bias_temp,
                                        src=col_rank+ddp_rank*model_parallel_size,
                                        group=mpu.get_summa_col_group())
        ctx.row_rank = row_rank
        ctx.col_rank = col_rank
        ctx.ddp_rank = ddp_rank
        ctx.model_parallel_size = model_parallel_size
        ctx.skip_bias_add = skip_bias_add
        if skip_bias_add:
            return bias_temp
        else:
            output = input + bias_temp
            return output

    @staticmethod
    def backward(ctx, output_grad):
        row_rank = ctx.row_rank
        col_rank = ctx.col_rank
        ddp_rank = ctx.ddp_rank
        model_parallel_size = ctx.model_parallel_size
        if ctx.skip_bias_add:
            torch.distributed.reduce(output_grad,
                                     col_rank+ddp_rank*model_parallel_size,
                                     group=mpu.get_summa_col_group())
            if row_rank == 0:
                return None, output_grad, None, None, None, None, None, None, None
            else:
                return None, None, None, None, None, None, None, None, None
        else:
            with torch.no_grad():
                reduce = torch.sum(output_grad, dim=(0, 1))
                torch.distributed.reduce(reduce,
                                         col_rank+ddp_rank*model_parallel_size,
                                         group=mpu.get_summa_col_group())
            if row_rank == 0:
                return output_grad, reduce, None, None, None, None, None, None, None
            else:
                return output_grad, None, None, None, None, None, None, None, None


class SUMMALinear(torch.nn.Module):
    def __init__(self, input_size, output_size, bias_flag=True,
                 init_method=init.xavier_normal_, skip_bias_add=False,
                 forward_buffer=None, backward_buffer=None,
                 parameter_gradient_buffer=None):
        super(SUMMALinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bias_flag = bias_flag
        self.skip_bias_add = skip_bias_add
        args = get_args()
        self.input_pp = mpu.divide(input_size, args.summa_dim)
        self.output_pp = mpu.divide(output_size, args.summa_dim)

        self.col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
        self.row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
        self.ddp_rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
        self.summa_dim = args.summa_dim
        self.model_parallel_size = args.model_parallel_size

        self.forward_buffer = forward_buffer
        self.backward_buffer = backward_buffer
        self.parameter_gradient_buffer = parameter_gradient_buffer

        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.input_pp,
                self.output_pp,
                dtype=args.params_dtype,
                device=torch.cuda.current_device()))
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.input_size, self.output_size,
                init_method, self.row_rank, self.col_rank)
        else:
            self.weight = Parameter(torch.empty(
                self.input_pp,
                self.output_pp,
                # device=torch.cuda.current_device(),
                dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method, stride=1)

        if bias_flag:
            if self.row_rank == 0:
                self.bias = Parameter(torch.empty(
                    self.output_pp,
                    device=torch.cuda.current_device(),
                    dtype=args.params_dtype))
            else:
                self.bias = torch.tensor(
                    0,dtype=args.params_dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=True)

    def forward(self, input):
        # input: [b/q, s, h/q]
        outshape = input.shape[:-1] + (self.output_pp, )
        output = SUMMA_AB.apply(
            input, self.weight, outshape,
            self.row_rank, self.col_rank, self.ddp_rank,
            self.summa_dim, self.model_parallel_size,
            self.forward_buffer, self.backward_buffer,
            self.parameter_gradient_buffer)
        if self.bias_flag:
            if self.skip_bias_add:
                bias = SUMMAbias.apply(
                    None, self.bias, self.output_pp,
                    self.row_rank, self.col_rank, self.ddp_rank,
                    self.model_parallel_size, True,
                    input.dtype)
                return output, bias
            else:
                output = SUMMAbias.apply(
                    output, self.bias, self.output_pp,
                    self.row_rank, self.col_rank, self.ddp_rank,
                    self.model_parallel_size, False,
                    input.dtype)
                return output
        else:
            return output