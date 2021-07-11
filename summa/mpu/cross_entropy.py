import torch
from .initialize import get_summa_row_group
from .initialize import get_summa_col_group
from .initialize import get_data_parallel_group

class SUMMA_CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _vocab_parallel_logits, target, vocab_start, vocab_end):
        # vocab_parallel_logits: [b/q, s, v/q]
        # target: [b/q, s]
        logits_max = torch.max(_vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(
            logits_max,
            op=torch.distributed.ReduceOp.MAX,
            group=get_summa_row_group())
        # Subtract the maximum value.
        # vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))
        vocab_parallel_logits = _vocab_parallel_logits - logits_max.unsqueeze(dim=-1)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start) | (target >= vocab_end)
        masked_target = target.clone() - vocab_start
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        # [b/q, s, v/q] -> [bs/q, v/q]
        logits_2d = vocab_parallel_logits.view(-1, vocab_end-vocab_start)
        # [b/q, s] -> [bs/q]
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0],
                                 device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_summa_row_group())

        # Sum of exponential of logits along vocab dimension across all GPUs.
        # [b/q, s, v/q]
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        # [b/q, s]
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_summa_row_group())

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Store softmax, target-mask and masked-target for backward pass.
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, output_grad):

        # Retreive tensors from the forward path.
        # [b/q, s, v/q], [b/q, s], [bs/q]
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        # [bs/q, v/q]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0],
                                 device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= (
                1.0 - target_mask.view(-1).float())

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(output_grad.unsqueeze(dim=-1))

        return grad_input, None, None, None