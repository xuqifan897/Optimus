import torch
from summa.mpu.layers import SUMMA_AB
from summa.mpu.layers import SUMMA_ABT
from summa.mpu.layers import SUMMA_ATB
import summa.mpu as mpu


class checkpoint_in_conjunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        conjunction = mpu.get_conjunction_gradient_buffer()
        conjunction.reset()
        output = conjunction.add(input)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad


class DENSE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, C_shape, row_rank, col_rank,
                ddp_rank, summa_dim, model_parallel_size):
        ctx.save_for_backward(A, B)
        ctx.row_rank = row_rank
        ctx.col_rank = col_rank
        ctx.ddp_rank = ddp_rank
        ctx.summa_dim = summa_dim
        ctx.model_parallel_size = model_parallel_size
        ctx.forward_buffer = mpu.get_QKV_forward_buffer()
        ctx.backward_buffer = mpu.get_conjunction_gradient_buffer()
        # ctx.forward_buffer = None
        # ctx.backward_buffer = None
        ctx.parameter_gradient_buffer = mpu.get_parameter_gradient_buffer()

        ctx.forward_buffer.reset()
        output = SUMMA_AB.apply(
            A, B, C_shape, row_rank, col_rank, ddp_rank,
            summa_dim, model_parallel_size, ctx.forward_buffer, None, None)

        return output

    @staticmethod
    def backward(ctx, output_grad):
        A, B = ctx.saved_tensors
        ctx.backward_buffer.reset()
        A_grad = SUMMA_ABT.apply(
            output_grad, B, A.shape, ctx.row_rank, ctx.col_rank,
            ctx.ddp_rank, ctx.summa_dim, ctx.model_parallel_size,
            ctx.backward_buffer, None, None)

        # dense_buffer = mpu.get_lmhead_dense_buffer()
        # dense_buffer.reset()
        # A_grad = dense_buffer.add(A_grad)

        B_grad = SUMMA_ATB.apply(
            A, output_grad, B.shape, ctx.row_rank, ctx.col_rank,
            ctx.ddp_rank, ctx.summa_dim, ctx.model_parallel_size,
            ctx.parameter_gradient_buffer, None, None)

        return A_grad, B_grad, None, None, None, None, None, None
