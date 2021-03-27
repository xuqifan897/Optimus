from summa.initialize import initialize_optimus
from summa.model.utils import init_method_normal, scaled_init_method_normal
from summa import get_args
from summa.mpu.layers import _initialize_affine_weight_gpu
import summa.mpu as mpu
import torch
import time

def main():
    args_defaults = {'tokenizer_type': 'BertWordPieceLowerCase'}
    extra_args_provider = None
    initialize_optimus(extra_args_provider=extra_args_provider,
                       args_defaults=args_defaults, allow_no_cuda=False)

    args = get_args()
    from summa.model_new.transformer import ParallelTransformer
    from summa.model_new.bert_model import bert_attention_mask_func
    init_method = init_method_normal(args.init_method_std)
    scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                   args.num_layers)
    layer = ParallelTransformer(
        bert_attention_mask_func, init_method, scaled_init_method)


    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in layer.parameters()])), flush=True)

    layer = layer.to(device=torch.cuda.current_device())

    batch_pp = mpu.divide(args.batch_size, args.summa_dim)
    hidden_pp = mpu.divide(args.hidden_size, args.summa_dim)
    size = (batch_pp, args.seq_length, hidden_pp)

    col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
    row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
    ddp_rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
    summa_dim = args.summa_dim
    model_parallel_size = args.model_parallel_size

    mask_size = (batch_pp, 1, args.seq_length, args.seq_length)
    mask = torch.empty(mask_size, dtype=args.params_dtype,
                       device=torch.cuda.current_device(),
                       requires_grad=False)
    _initialize_affine_weight_gpu(mask, init_method)
    torch.distributed.broadcast(
        mask,
        src=row_rank*summa_dim+ddp_rank*model_parallel_size,
        group=mpu.get_summa_row_group())

    grad = torch.empty(size, dtype=args.params_dtype,
                         device=torch.cuda.current_device(),
                         requires_grad=False)
    _initialize_affine_weight_gpu(grad, init_method)

    time_forward = []
    time_backward = []
    input = torch.empty(size, dtype=args.params_dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False)
    for i in range(args.eval_iters):
        if args.rank == 0:
            print('step start: {}'.format(i), flush=True)
        _initialize_affine_weight_gpu(input, init_method)
        input.requires_grad = True

        time_start = time.time()
        output = layer(input, mask)
        time_1 = time.time()
        output.backward(grad)
        time_2 = time.time()
        time_forward.append(time_1 - time_start)
        time_backward.append(time_2 - time_1)
        if args.rank == 0:
            print('step end: {}'.format(i), flush=True)

    time_forward_avg = sum(time_forward[1:])/(args.eval_iters - 1)
    time_backward_avg = sum(time_backward[1:])/(args.eval_iters - 1)
    if args.rank == 0:
        print('average forward time: {}'.format(time_forward_avg), flush=True)
        print('average backward time: {}\n\n'.format(time_backward_avg), flush=True)
        print('forward time:')
        print(time_forward, '\n\n')
        print('backward time:')
        print(time_backward)

if __name__ == '__main__':
    main()