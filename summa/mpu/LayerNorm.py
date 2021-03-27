import torch
from summa import get_args
import summa.mpu as mpu
from torch.nn.parameter import Parameter
from summa.mpu.layers import SUMMAbias

class LayerNorm_summa(torch.nn.Module):
    def __init__(self, dim):
        super(LayerNorm_summa, self).__init__()
        # By default, this module normalize along the last dimension
        args = get_args()
        self.col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
        self.row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
        self.ddp_rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
        self.summa_dim = args.summa_dim
        self.model_parallel_size = args.model_parallel_size
        self.hidden_size = args.hidden_size
        self.dim = dim
        self.dtype = args.params_dtype
        if self.row_rank == 0:
            self.gamma = Parameter(torch.ones(
                dim,
                # device=torch.cuda.current_device(),
                dtype=args.params_dtype))
            self.beta = Parameter(torch.zeros(
                dim,
                # device=torch.cuda.current_device(),
                dtype=args.params_dtype))
        else:
            self.gamma = Parameter(torch.tensor(
                1,
                dtype=args.params_dtype,
                device=torch.cuda.current_device(),
                requires_grad=True))
            self.beta = Parameter(torch.tensor(
                1,
                dtype=args.params_dtype,
                device=torch.cuda.current_device(),
                requires_grad=True))
        self.epsilon = torch.tensor(
            args.layernorm_epsilon,
            device=torch.cuda.current_device(),
            dtype=args.params_dtype,
            requires_grad=False)

    def forward(self, input):
        with torch.no_grad():
            E_x = torch.sum(input, dim=-1, keepdim=True) # [b/q, s, 1]
            torch.distributed.all_reduce(E_x, group=mpu.get_summa_row_group())
            E_x /= self.hidden_size

            # Var_x in the block below is the sum of input^2
            Var_x = torch.sum(input*input, dim=-1, keepdim=True) # [b/q, s, 1]
            torch.distributed.all_reduce(Var_x, group=mpu.get_summa_row_group())
            Var_x /= self.hidden_size

            Var_x = Var_x - E_x * E_x # variance of x [b/q, s, 1]
            Var_x = 1.0 / torch.sqrt(Var_x + self.epsilon)  # this time 1/sqrt(Var_x + epsilon)

        output = ParallelLayerNorm.apply(input, E_x, Var_x, self.hidden_size)
        bias = SUMMAbias.apply(
            None, self.beta, self.dim, self.row_rank,
            self.col_rank, self.ddp_rank, self.model_parallel_size,
            True, output.dtype)
        scale = SUMMAbias.apply(
            None, self.gamma, self.dim, self.row_rank,
            self.col_rank, self.ddp_rank, self.model_parallel_size,
            True, output.dtype)
        output = torch.addcmul(bias, scale, output)
        # output = AffineMul.apply(
        #     output, self.gamma, self.row_rank, self.col_rank,
        #     self.ddp_rank, self.model_parallel_size, self.dim, self.dtype)
        # output = SUMMAbias.apply(
        #     output, self.beta, self.dim, self.row_rank,
        #     self.col_rank, self.ddp_rank, self.model_parallel_size,
        #     False, output.dtype)
        return output


class ParallelLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, E_x, Var_x, hidden_size):
        input = input - E_x
        # in here, input = x - E[x], Var_x = 1 / sqrt(Var[x] + eps)
        ctx.hidden_size = hidden_size
        output = input * Var_x
        ctx.save_for_backward(output, Var_x)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        x, Var_x = ctx.saved_tensors
        # in here, Var_x = 1 / sqrt(Var[x] + eps), x = (x - E[x]) * Var_x
        with torch.no_grad():
            output_grad_sum = torch.sum(output_grad, dim=-1, keepdim=True)
            torch.distributed.all_reduce(
                output_grad_sum, group=mpu.get_summa_row_group())
            output_grad_sum /= ctx.hidden_size

            output_grad_mul_x_sum = torch.sum(output_grad*x, dim=-1, keepdim=True)
            torch.distributed.all_reduce(
                output_grad_mul_x_sum, group=mpu.get_summa_row_group())
            output_grad_mul_x_sum /= ctx.hidden_size

            input_grad = output_grad.clone()
            input_grad -= x * output_grad_mul_x_sum
            input_grad -= output_grad_sum
            input_grad *= Var_x

        return input_grad, None, None, None


class AffineMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma, row_rank, col_rank,
                ddp_rank, model_parallel_size, dim, dtype):
        with torch.no_grad():
            if row_rank == 0:
                gamma_temp = gamma.clone()
            else:
                gamma_temp = torch.zeros(dim,
                                         dtype=dtype,
                                         device=torch.cuda.current_device())
            torch.distributed.broadcast(gamma_temp,
                                        src=col_rank+ddp_rank*model_parallel_size,
                                        group=mpu.get_summa_col_group())
            output = input * gamma_temp
        ctx.save_for_backward(input, gamma_temp)
        ctx.row_rank = row_rank
        ctx.col_rank = col_rank
        ctx.ddp_rank = ddp_rank
        ctx.model_parallel_size = model_parallel_size
        return output

    @staticmethod
    def backward(ctx, output_grad):
        input, gamma_temp = ctx.saved_tensors
        input_grad = output_grad * gamma_temp
        gamma_grad = torch.sum(output_grad * input, dim=[0, 1])
        torch.distributed.reduce(gamma_grad,
                                 dst=ctx.col_rank+ctx.ddp_rank*ctx.model_parallel_size,
                                 group=mpu.get_summa_col_group())
        if ctx.row_rank == 0:
            return input_grad, gamma_grad, None, None, None, None, None, None
        else:
            return input_grad, None, None, None, None, None, None, None