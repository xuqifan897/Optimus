from summa.initialize import initialize_optimus
from summa import get_args
from summa import print_rank_0
from summa import get_timers
from summa import mpu
from summa.data.samplers import DistributedBatchSampler
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from summa.model import DistributedDataParallel as LocalDDP
import torch
import torch.nn.functional as F
from summa.model.utils import init_method_normal, scaled_init_method_normal

def master_and_block(size, summa_dim, row_rank,
                     col_rank, device, requires_grad):
    args = get_args()
    grad_master = torch.rand(size, device=device, requires_grad=requires_grad)
    with torch.no_grad():
        grad_master = grad_master.to(dtype=args.params_dtype)
    grad_master.requires_grad = requires_grad
    grad_row_split = torch.split(grad_master, size[0]//summa_dim, dim=0)
    grad_row = grad_row_split[row_rank]
    grad_col_split = torch.split(grad_row, size[-1]//summa_dim, dim=-1)
    grad = grad_col_split[col_rank]
    with torch.no_grad():
        grad = grad.clone()
    grad.requires_grad = requires_grad
    return grad_master, grad


def col_block(size, summa_dim, col_rank, device, requires_grad):
    args = get_args()
    grad_master = torch.randn(
        size, device=device, requires_grad=requires_grad)
    with torch.no_grad():
        grad_master = grad_master.to(dtype=args.params_dtype)
    grad_master.requires_grad = requires_grad
    grad_col_split = torch.split(grad_master, size[-1]//summa_dim, dim=-1)
    grad = grad_col_split[col_rank]
    with torch.no_grad():
        grad = grad.clone()
    grad.requires_grad = requires_grad
    return grad_master, grad


def tst_VocabParallelEmbedding(tokens):
    args = get_args()
    init_method = init_method_normal(args.init_method_std)
    from summa.mpu.layers import VocabParallelEmbedding
    layer = VocabParallelEmbedding(
        args.padded_vocab_size, args.hidden_size, init_method)
    weight_master_ = layer.weight_master
    with torch.no_grad():
        weight_master = weight_master_.clone()
    weight_master.requires_grad = True

    out = layer(tokens)
    out_master = F.embedding(tokens, weight_master)

    col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
    row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
    size = (args.batch_size, args.seq_length, args.hidden_size)
    grad_master, grad = master_and_block(
        size, args.summa_dim, row_rank, col_rank,
        torch.cuda.current_device(), False)
    out_master.backward(grad_master)
    out.backward(grad)
    import pdb; pdb.set_trace()
    print('haha')


def tst_PosParallelEmbedding():
    args = get_args()
    col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
    row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
    ddp_rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
    size = (args.batch_size, args.seq_length, args.hidden_size)
    input_master, input = master_and_block(
        size, args.summa_dim, row_rank, col_rank,
        torch.cuda.current_device(), True)
    size_weight = (args.seq_length, args.hidden_size)
    weight_master, weight = col_block(
        size_weight, args.summa_dim, col_rank,
        torch.cuda.current_device(), True)
    init_method = init_method_normal(args.init_method_std)
    from summa.mpu.layers import PosParallelEmbedding
    layer = PosParallelEmbedding(
        args.seq_length, args.hidden_size, init_method)
    with torch.no_grad():
        if row_rank == 0:
            layer.weight.copy_(weight)
    output = layer(input)
    output_master = input_master + weight_master

    grad_master, grad = master_and_block(
        size, args.summa_dim, row_rank, col_rank,
        torch.cuda.current_device(), True)
    output.backward(grad)
    output_master.backward(grad_master)
    import pdb; pdb.set_trace()
    print('haha')


def tst_TokentypeParallelEmbedding(types):
    args = get_args()
    col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
    row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
    init_method = init_method_normal(args.init_method_std)

    from summa.mpu.layers import TokentypeParallelEmbedding
    layer = TokentypeParallelEmbedding(2, args.hidden_size, init_method)
    weight_size = (2, args.hidden_size)
    weight_master, weight = col_block(
        weight_size, args.summa_dim, col_rank,
        torch.cuda.current_device(), True)
    if row_rank == 0:
        with torch.no_grad():
            layer.weight.copy_(weight)
    out = layer(types)
    out_master = F.embedding(types, weight_master)
    size = (types.size(0), types.size(1), args.hidden_size)
    grad_master, grad = master_and_block(
        size, args.summa_dim, row_rank, col_rank,
        torch.cuda.current_device(), False)
    out.backward(grad)
    out_master.backward(grad_master)
    import pdb; pdb.set_trace()
    print('haha')


def tst_Embedding(input_ids, position_ids, tokentype_ids):
    args = get_args()
    from summa.model_new.language_model import Embedding
    init_method = init_method_normal(args.init_method_std)
    layer = Embedding(args.hidden_size, args.padded_vocab_size,
                      args.seq_length, args.hidden_dropout, init_method,
                      num_tokentypes=2)
    hidden = layer(input_ids, position_ids, tokentype_ids)
    state_dict = layer.state_dict_for_save_checkpoint()
    layer.load_state_dict(state_dict)
    print('haha')


def tst_SUMMA_AB():
    args = get_args()
    col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
    row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
    ddp_rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
    A_shape = (args.batch_size, args.seq_length, args.hidden_size)
    B_shape = (args.hidden_size, 4*args.hidden_size)
    C_shape = (args.batch_size, args.seq_length, 4*args.hidden_size)
    A_master, A = master_and_block(
        A_shape, args.summa_dim, row_rank, col_rank,
        torch.cuda.current_device(), True)
    B_master, B = master_and_block(
        B_shape, args.summa_dim, row_rank, col_rank,
        torch.cuda.current_device(), True)
    grad_master, grad = master_and_block(
        C_shape, args.summa_dim, row_rank, col_rank,
        torch.cuda.current_device(), False)
    from summa.mpu.layers import SUMMA_AB
    C = SUMMA_AB.apply(
        A, B, grad.shape, row_rank, col_rank, ddp_rank,
        args.summa_dim, args.model_parallel_size)
    C_master = torch.matmul(A_master, B_master)
    C.backward(grad)
    C_master.backward(grad_master)
    import pdb; pdb.set_trace()
    print('haha')


def tst_SUMMA_ABT():
    args = get_args()
    col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
    row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
    ddp_rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
    A_shape = (args.batch_size, args.seq_length, 4*args.hidden_size)
    B_shape = (args.hidden_size, 4*args.hidden_size)
    C_shape = (args.batch_size, args.seq_length, args.hidden_size)
    A_master, A = master_and_block(
        A_shape, args.summa_dim, row_rank, col_rank,
        torch.cuda.current_device(), True)
    B_master, B = master_and_block(
        B_shape, args.summa_dim, row_rank, col_rank,
        torch.cuda.current_device(), True)
    grad_master, grad = master_and_block(
        C_shape, args.summa_dim, row_rank, col_rank,
        torch.cuda.current_device(), False)
    from summa.mpu.layers import SUMMA_ABT
    C = SUMMA_ABT.apply(
        A, B, grad.shape, row_rank, col_rank, ddp_rank,
        args.summa_dim, args.model_parallel_size)
    C_master = torch.matmul(A_master, B_master.transpose(0, 1))
    C.backward(grad)
    C_master.backward(grad_master)
    import pdb; pdb.set_trace()
    print('haha')


def tst_SUMMA_ATB():
    args = get_args()
    col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
    row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
    ddp_rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
    A_shape = (4*args.hidden_size, args.hidden_size)
    B_shape = (4*args.hidden_size, args.hidden_size)
    C_shape = (args.hidden_size, args.hidden_size)
    A_master, A = master_and_block(
        A_shape, args.summa_dim, row_rank, col_rank,
        torch.cuda.current_device(), True)
    B_master, B = master_and_block(
        B_shape, args.summa_dim, row_rank, col_rank,
        torch.cuda.current_device(), True)
    grad_master, grad = master_and_block(
        C_shape, args.summa_dim, row_rank, col_rank,
        torch.cuda.current_device(), False)

    from summa.mpu.layers import SUMMA_ATB
    C = SUMMA_ATB.apply(
        A, B, grad.shape, row_rank, col_rank, ddp_rank,
        args.summa_dim, args.model_parallel_size)
    C_master = torch.matmul(A_master.transpose(0, 1), B_master)
    C.backward(grad)
    C_master.backward(grad_master)
    import pdb; pdb.set_trace()
    print('haha')


def tst_SUMMALinear():
    args = get_args()
    col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
    row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
    ddp_rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
    forward_buffer = mpu.get_forward_buffer()
    backward_buffer = mpu.get_backward_buffer()
    parameter_gradient_buffer = mpu.get_parameter_gradient_buffer()
    from summa.mpu import SUMMALinear
    init_method = init_method_normal(args.init_method_std)
    layer = SUMMALinear(
        args.hidden_size, 4*args.hidden_size, bias_flag=False,
        init_method=init_method, skip_bias_add=True,
        forward_buffer=forward_buffer, backward_buffer=backward_buffer,
        parameter_gradient_buffer=parameter_gradient_buffer)

    # bias_master = torch.rand(
    #     4*args.hidden_size, dtype=args.params_dtype,
    #     device=torch.cuda.current_device(),requires_grad=True)
    # hidden_pp = mpu.divide(4*args.hidden_size, args.summa_dim)
    # if row_rank == 0:
    #     bias_list = torch.split(bias_master, hidden_pp, dim=0)
    #     bias = bias_list[col_rank]
    #     with torch.no_grad():
    #         layer.bias.copy_(bias)
    #     layer.bias.requires_grad = True

    input_size = (args.batch_size, args.seq_length, args.hidden_size)
    output_size = (args.batch_size, args.seq_length, 4*args.hidden_size)
    input_master, input = master_and_block(
        input_size, args.summa_dim, row_rank,
        col_rank, torch.cuda.current_device(), True)
    grad_master, grad = master_and_block(
        output_size, args.summa_dim, row_rank,
        col_rank, torch.cuda.current_device(), False)
    # layer.master_weight.requires_grad = True
    # output, bias = layer(input)
    # output = output + bias
    output = layer(input)
    # output_master = torch.matmul(input_master, layer.master_weight) + bias_master
    import pdb; pdb.set_trace()
    output.backward(grad)
    # output_master.backward(grad_master)
    print('haha')


def tst_LayerNorm():
    args = get_args()
    col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
    row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
    input_size = (args.batch_size, args.seq_length, args.hidden_size)
    input_master, input = master_and_block(
        input_size, args.summa_dim, row_rank, col_rank,
        device=torch.cuda.current_device(), requires_grad=True)

    grad_master, grad = master_and_block(
        input_size, args.summa_dim, row_rank, col_rank,
        device=torch.cuda.current_device(), requires_grad=False)

    E_x = torch.mean(input_master, dim=-1, keepdim=True) # [b, ss, 1]
    Var_x = torch.var(input_master, dim=-1, keepdim=True, unbiased=False) # [b, ss, 1]
    x = (input_master - E_x) / torch.sqrt(Var_x + args.layernorm_epsilon) # [b, ss, hs]
    x.backward(grad_master)

    from summa.mpu.LayerNorm import LayerNorm_summa
    layer = LayerNorm_summa(512)
    out = layer(input)
    out.backward(grad)
    print('input.grad')


def tst_ParallelSelfAttention():
    args = get_args()
    from summa.model_new.transformer import ParallelSelfAttention
    from summa.model_new.bert_model import bert_attention_mask_func
    init_method = init_method_normal(args.init_method_std)
    scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                   args.num_layers)
    layer = ParallelSelfAttention(
        bert_attention_mask_func, init_method, scaled_init_method, 2)

    input_size = (args.batch_size, args.seq_length, args.hidden_size)
    col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
    row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
    _, input = master_and_block(
        input_size, args.summa_dim, row_rank, col_rank,
        device=torch.cuda.current_device(), requires_grad=True)
    attention_mask = torch.rand((
        mpu.divide(args.batch_size, args.summa_dim),
        1, args.seq_length, args.seq_length),
        device=torch.cuda.current_device(),
        requires_grad=False)
    _, grad = master_and_block(
        input_size, args.summa_dim, row_rank, col_rank,
        device=torch.cuda.current_device(), requires_grad=False)

    output = layer(input, attention_mask)
    output.backward(grad)
    print('haha')


def tst_ParallelMLP():
    args = get_args()
    row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
    col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
    input_size = (args.batch_size, args.seq_length, args.hidden_size)
    input_master, input = master_and_block(
        input_size, args.summa_dim, row_rank, col_rank,
        device=torch.cuda.current_device(), requires_grad=True)
    grad_master, grad = master_and_block(
        input_size, args.summa_dim, row_rank, col_rank,
        device=torch.cuda.current_device(), requires_grad=False)
    from summa.model_new.transformer import ParallelMLP
    init_method = init_method_normal(args.init_method_std)
    scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                   args.num_layers)
    layer = ParallelMLP(init_method, scaled_init_method)
    output = layer(input)
    A = layer.dense_h_to_4h.master_weight
    B = layer.dense_4h_to_h.master_weight
    A.requires_grad = True
    B.requires_grad = True
    output_master = torch.matmul(input_master, A)
    output_master = layer.activation_func(output_master)
    output_master = torch.matmul(output_master, B)

    output.backward(grad)
    output_master.backward(grad_master)
    import pdb; pdb.set_trace()
    print('haha')


def tst_ParallelTransformerLayer():
    args = get_args()
    col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
    row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
    input_size = (args.batch_size, args.seq_length, args.hidden_size)
    input_master, input = master_and_block(
        input_size, args.summa_dim, row_rank, col_rank,
        device=torch.cuda.current_device(), requires_grad=True)
    grad_master, grad = master_and_block(
        input_size, args.summa_dim, row_rank, col_rank,
        device=torch.cuda.current_device(), requires_grad=False)
    from summa.model_new.transformer import ParallelTransformerLayer
    from summa.model_new.bert_model import bert_attention_mask_func
    init_method = init_method_normal(args.init_method_std)
    scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                   args.num_layers)
    layer = ParallelTransformerLayer(
        bert_attention_mask_func, init_method,
        scaled_init_method, args.num_layers)

    attention_mask = torch.rand((
        mpu.divide(args.batch_size, args.summa_dim),
        1, args.seq_length, args.seq_length),
        device=torch.cuda.current_device(),
        requires_grad=False)

    output = layer(input, attention_mask)
    import pdb; pdb.set_trace()
    output.backward(grad)
    print('haha')


def tst_ParallelTransformer():
    args = get_args()
    from summa.model_new.transformer import ParallelTransformer
    from summa.model_new.bert_model import bert_attention_mask_func
    col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
    row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
    init_method = init_method_normal(args.init_method_std)
    scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                   args.num_layers)
    layer = ParallelTransformer(
        bert_attention_mask_func, init_method, scaled_init_method)

    size = (args.batch_size, args.seq_length, args.hidden_size)
    input_master, input = master_and_block(
        size, args.summa_dim, row_rank, col_rank,
        device=torch.cuda.current_device(), requires_grad = True)
    grad_master, grad = master_and_block(
        size, args.summa_dim, row_rank, col_rank,
        device=torch.cuda.current_device(), requires_grad = False)
    attention_mask = torch.rand((
        mpu.divide(args.batch_size, args.summa_dim),
        1, args.seq_length, args.seq_length),
        device=torch.cuda.current_device(),
        requires_grad=False)
    output = layer(input, attention_mask)
    output.backward(grad)
    import pdb; pdb.set_trace()
    print('haha')


def tst_SUMMA_BinaryHead():
    args = get_args()
    col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
    row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
    input_size = (args.batch_size, args.seq_length, args.hidden_size)
    input_master, input = master_and_block(
        input_size, args.summa_dim, row_rank, col_rank,
        device=torch.cuda.current_device(), requires_grad=True)

    grad_size = (args.batch_size, args.seq_length, 2)
    grad_master = torch.randn(
        grad_size,
        dtype=args.params_dtype,
        device=torch.cuda.current_device())
    batch_pp = mpu.divide(args.batch_size, args.summa_dim)
    grad_list = torch.split(grad_master, batch_pp)
    grad = grad_list[row_rank]

    weight_master = torch.randn(
        (args.hidden_size, 2),
        dtype=args.params_dtype,
        device=torch.cuda.current_device(),
        requires_grad=True)

    from summa.model_new.bert_model import BinaryHead
    init_method = init_method_normal(args.init_method_std)
    layer = BinaryHead(args.hidden_size, 2, init_method)

    hidden_pp = mpu.divide(args.hidden_size, args.summa_dim)
    if row_rank == 0:
        weight_list = torch.split(weight_master, hidden_pp, dim=0)
        weight = weight_list[col_rank]
        with torch.no_grad():
            layer.weight.copy_(weight)
        layer.weight.requires_grad = True

    output = layer(input)
    output_master = torch.matmul(input_master, weight_master)

    output.backward(grad)
    output_master.backward(grad_master)
    import pdb; pdb.set_trace()
    print('haha')


def tst_SUMMA_CrossEntropy():
    args = get_args()
    col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
    row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
    vocab_pp = mpu.divide(args.padded_vocab_size, args.summa_dim)
    batch_pp = mpu.divide(args.batch_size, args.summa_dim)
    vocab_start = vocab_pp * col_rank
    vocab_end = vocab_start + vocab_pp
    logits_shape = (args.batch_size, args.seq_length, args.padded_vocab_size)
    logits_master, logits = master_and_block(
        logits_shape, args.summa_dim, row_rank, col_rank,
        device=torch.cuda.current_device(), requires_grad = True)

    target_shape = (args.batch_size, args.seq_length)
    target_master = torch.rand(
        target_shape,
        dtype=torch.double,
        device=torch.cuda.current_device())\
                    * args.padded_vocab_size
    target_master = target_master.type(torch.cuda.LongTensor)

    target_list = torch.split(target_master, batch_pp, dim=0)
    target = target_list[row_rank]

    output = mpu.SUMMA_CrossEntropy.apply(
        logits, target, vocab_start, vocab_end)

    # [b, s, v]
    output_master = torch.exp(logits_master).view((-1, args.padded_vocab_size))
    sum_exp_logits = torch.sum(output_master, dim=-1)
    arange_1d = torch.arange(
        start=0, end=output_master.shape[0], device=output_master.device)
    logits_2d = logits_master.view((-1, args.padded_vocab_size))
    output_master = torch.log(sum_exp_logits) - logits_2d[arange_1d, target_master.view(-1)]
    output_master = output_master.view(target_shape)

    grad_master = torch.rand(
        target_shape,
        device=torch.cuda.current_device())
    grad_list = torch.split(grad_master, batch_pp, dim=0)
    grad = grad_list[row_rank]

    output.backward(grad)
    output_master.backward(grad_master)
    import pdb; pdb.set_trace()
    print('haha')


def tst_BertModel(tokens, loss_mask, types, lm_labels):
    args = get_args()
    col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
    row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
    ddp_rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
    from summa.model_new.bert_model import BertModel
    model = BertModel(2, True)
    batch_pp = mpu.divide(args.batch_size, args.summa_dim)
    loss_mask_list = torch.split(loss_mask, batch_pp, dim=0)
    attention_mask = loss_mask_list[row_rank]

    lm_labels_list = torch.split(lm_labels, batch_pp, dim=0)
    lm_labels = lm_labels_list[row_rank]
    lm_loss, binary_logits = model(tokens, attention_mask, types, lm_labels)
    loss = torch.sum(lm_loss) + torch.sum(binary_logits)
    loss.backward()
    import pdb; pdb.set_trace()
    print('haha')


def tst_TransformerLanguageModel():
    args = get_args()
    col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
    row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
    from summa.model_new.bert_model import bert_attention_mask_func
    init_method = init_method_normal(args.init_method_std)
    scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                   args.num_layers)
    from summa.model_new.language_model import get_language_model
    model, _ = get_language_model(
        bert_attention_mask_func, 2, True,
        init_method, scaled_init_method)
    target_shape = (args.batch_size, args.seq_length)
    tokens = torch.rand(
        target_shape,
        dtype=torch.double,
        device=torch.cuda.current_device()) \
                    * args.padded_vocab_size
    tokens = tokens.type(torch.cuda.LongTensor)

    tokentypes = torch.rand(
        target_shape,
        dtype=torch.double,
        device=torch.cuda.current_device()) * 2
    tokentypes = tokentypes.type(torch.cuda.LongTensor)

    batch_pp = mpu.divide(args.batch_size, args.summa_dim)
    attention_mask_shape = (batch_pp, 1, args.seq_length, args.seq_length)
    attention_mask = torch.rand(
        attention_mask_shape,
        dtype=torch.float32,
        device=torch.cuda.current_device())

    transformer_output, pooled_output = model(
        tokens, None, attention_mask, tokentypes)
    loss = torch.sum(transformer_output) + torch.sum(pooled_output)
    loss.backward()
    print('haha')


def tst_Pooler():
    args = get_args()
    init_method = init_method_normal(args.init_method_std)
    from summa.model_new.language_model import Pooler
    layer = Pooler(args.hidden_size, init_method)
    input_size = (args.batch_size, args.seq_length, args.hidden_size)
    col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
    row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
    input_master, input = master_and_block(
        input_size, args.summa_dim, row_rank, col_rank,
        device=torch.cuda.current_device(), requires_grad=True)
    output = layer(input, 0)
    output = torch.sum(output)
    output.backward()
    print('haha')


def broadcast_column():
    args = get_args()
    assert args.world_size == 8
    col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
    row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
    ddp_rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
    num_iters = 16
    matrix_size = (1024, 1024, 1024)
    A = torch.empty(
        matrix_size,
        dtype=torch.float32,
        device=torch.cuda.current_device())
    init_method = init_method_normal(args.init_method_std)
    from summa.mpu.layers import _initialize_affine_weight_gpu
    import time
    if args.rank == 0:
        print('reaches here!', flush=True)

    times = []
    for i in range(num_iters):
        src = (args.rank % args.model_parallel_size)
        _initialize_affine_weight_gpu(A, init_method)
        torch.distributed.barrier()
        time_start = time.time()
        torch.distributed.broadcast(
            A,
            src=src,
            group=mpu.get_data_parallel_group())
        torch.distributed.barrier()
        time_end = time.time()
        times.append(time_end - time_start)
        if args.rank == 0:
            print('time: {}'.format(times[-1]), flush=True)

    time = sum(times[1:]) / (num_iters - 1)
    if args.rank == 0:
        print('\naverage time: {}'.format(time))


def Compare_broadcast_AllReduce():
    args = get_args()
    assert args.world_size == 8
    assert args.model_parallel_size == 4
    col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
    row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
    ddp_rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())

    num_iters = 16
    matrix_size = (1024, 1024, 1024)
    A = torch.empty(
        matrix_size,
        dtype=torch.float32,
        device=torch.cuda.current_device())
    init_method = init_method_normal(args.init_method_std)
    from summa.mpu.layers import _initialize_affine_weight_gpu
    import time
    if args.rank == 0:
        print('reaches here!', flush=True)

    # a group of size 2
    world_2_broadcast = []
    world_2_AllReduce = []
    for i in range(num_iters):
        src = (i % args.summa_dim) + row_rank * args.summa_dim\
              + ddp_rank * args.model_parallel_size
        _initialize_affine_weight_gpu(A, init_method)
        torch.distributed.barrier()
        time_start = time.time()
        torch.distributed.broadcast(
            A, src=src,
            group=mpu.get_summa_row_group())
        torch.distributed.barrier()
        time_end = time.time()
        world_2_broadcast.append(time_end - time_start)
        if args.rank == 0:
            print('broadcast time: {}'.format(world_2_broadcast[-1]), flush=True)

        _initialize_affine_weight_gpu(A, init_method)
        torch.distributed.barrier()
        time_start = time.time()
        torch.distributed.all_reduce(A, group=mpu.get_summa_row_group())
        torch.distributed.barrier()
        time_end = time.time()
        world_2_AllReduce.append(time_end - time_start)
        if args.rank == 0:
            print('AllReduce time: {}'.format(world_2_AllReduce[-1]), flush=True)

    # a group of size 4
    world_4_broadcast = []
    world_4_AllReduce = []
    for i in range(num_iters):
        src = (i % args.model_parallel_size) \
              + ddp_rank * args.model_parallel_size
        _initialize_affine_weight_gpu(A, init_method)
        torch.distributed.barrier()
        time_start = time.time()
        torch.distributed.broadcast(
            A, src=src,
            group=mpu.get_model_parallel_group())
        torch.distributed.barrier()
        time_end = time.time()
        world_4_broadcast.append(time_end - time_start)
        if args.rank == 0:
            print('broadcast time: {}'.format(world_4_broadcast[-1]), flush=True)

        _initialize_affine_weight_gpu(A, init_method)
        torch.distributed.barrier()
        time_start = time.time()
        torch.distributed.all_reduce(A, group=mpu.get_model_parallel_group())
        torch.distributed.barrier()
        time_end = time.time()
        world_4_AllReduce.append(time_end - time_start)
        if args.rank == 0:
            print('AllReduce time: {}'.format(world_4_AllReduce[-1]), flush=True)

    # a group of size 4
    world_8_broadcast = []
    world_8_AllReduce = []
    for i in range(num_iters):
        src = i % args.world_size
        _initialize_affine_weight_gpu(A, init_method)
        torch.distributed.barrier()
        time_start = time.time()
        torch.distributed.broadcast(
            A, src=src)
        torch.distributed.barrier()
        time_end = time.time()
        world_8_broadcast.append(time_end - time_start)
        if args.rank == 0:
            print('broadcast time: {}'.format(world_8_broadcast[-1]), flush=True)

        _initialize_affine_weight_gpu(A, init_method)
        torch.distributed.barrier()
        time_start = time.time()
        torch.distributed.all_reduce(A)
        torch.distributed.barrier()
        time_end = time.time()
        world_8_AllReduce.append(time_end - time_start)
        if args.rank == 0:
            print('AllReduce time: {}'.format(world_8_AllReduce[-1]), flush=True)

    if args.rank == 0:
        print('world 2 broadcast: {}'.format(sum(world_2_broadcast[1:])/(num_iters-1)), flush=True)
        print('world 2 AllReduce: {}'.format(sum(world_2_AllReduce[1:])/(num_iters-1)), flush=True)
        print('world 4 broadcast: {}'.format(sum(world_4_broadcast[1:])/(num_iters-1)), flush=True)
        print('world 4 AllReduce: {}'.format(sum(world_4_AllReduce[1:])/(num_iters-1)), flush=True)
        print('world 8 broadcast: {}'.format(sum(world_8_broadcast[1:])/(num_iters-1)), flush=True)
        print('world 8 AllReduce: {}'.format(sum(world_8_AllReduce[1:])/(num_iters-1)), flush=True)


def tst_bias_dropout_add():
    from summa.model_new.transformer import bias_dropout_add_fused_train
    x = torch.rand((4, 4, 4), dtype=torch.float32,
                   device=torch.cuda.current_device(),
                   requires_grad=True)
    bias = torch.rand((1, 4), dtype=torch.float32,
                      device=torch.cuda.current_device(),
                      requires_grad=True)
    residual = torch.rand((4, 4, 4), dtype=torch.float32,
                          device=torch.cuda.current_device(),
                          requires_grad=True)
    out = bias_dropout_add_fused_train(x, bias, residual, 0.1)
    out = torch.sum(out)
    out.backward()
    import pdb; pdb.set_trace()
    print('haha')



def Develop(train_valid_test_dataset_provider, model_provider,
            extra_args_provider=None, args_defaults={}):

    initialize_optimus(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults, allow_no_cuda=False)

    # args = get_args()
    # timers = get_timers()
    #
    # timers('train/valid/test data iterators').start()
    # train_data_iterator, valid_data_iterator, test_data_iterator \
    #     = build_train_valid_test_data_iterators(
    #     train_valid_test_dataset_provider)
    # timers('train/valid/test data iterators').stop()
    #
    # tokens, types, sentence_order, loss_mask, lm_labels, padding_mask \
    #     = get_batch(train_data_iterator)

    # tst_VocabParallelEmbedding(tokens)
    # tst_PosParallelEmbedding()
    # tst_TokentypeParallelEmbedding(types)
    # tst_SUMMA_AB()
    # tst_SUMMA_ABT()
    # tst_SUMMA_ATB()
    # tst_LayerNorm()
    # tst_Embedding(tokens, None, types)
    # tst_SUMMALinear()
    # tst_ParallelSelfAttention()
    # tst_ParallelMLP()
    # tst_ParallelTransformerLayer()
    tst_ParallelTransformer()
    # tst_SUMMA_BinaryHead()
    # tst_SUMMA_CrossEntropy()
    # tst_BertModel(tokens, loss_mask, types, lm_labels)
    # tst_TransformerLanguageModel()
    # tst_Pooler()
    # Compare_broadcast_AllReduce()
    # broadcast_column()
    # tst_bias_dropout_add()


def get_batch(data_iterator):
    """Build the batch."""

    # Items and their type.
    keys = ['text', 'types', 'labels', 'is_random', 'loss_mask', 'padding_mask']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b['text'].long()
    types = data_b['types'].long()
    sentence_order = data_b['is_random'].long()
    loss_mask = data_b['loss_mask'].float()
    lm_labels = data_b['labels'].long()
    padding_mask = data_b['padding_mask'].long()

    return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask


def get_model(model_provider_func):
    """Build the model."""
    args = get_args()

    # Build model on cpu.
    model = model_provider_func()

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # # Fp16 conversion.
    # if args.fp16:
    #     model = FP16_Module(model)

    # Wrap model for distributed training."""
    if args.DDP_impl == 'torch':
        i = torch.cuda.current_device()
        model = torchDDP(model, device_ids=[i], output_device=i,
                         process_group=mpu.get_data_parallel_group())
        return model
    if args.DDP_impl == 'local':
        model = LocalDDP(model)
        return model

    raise NotImplementedError('Unknown DDP implementation specified: {}. '
                              'Exiting.'.format(args.DDP_impl))


def make_data_loader(dataset):
    """Buld dataloader given an input dataset."""
    if dataset is None:
        return None
    args = get_args()

    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = args.batch_size * world_size
    num_workers = args.num_workers

    # Use a simple sampler with distributed batch sampler.
    sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(sampler=sampler,
                                            batch_size=global_batch_size,
                                            drop_last=True,
                                            rank=rank,
                                            world_size=world_size)
    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True)


def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider):
    """XXX"""
    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)
    print_rank_0('> building train, validation, and test datasets ...')
    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0:
        # Rank, size, and global batch size.
        data_parallel_size = mpu.get_data_parallel_world_size()
        global_batch_size = args.batch_size * data_parallel_size

        # Number of train/valid/test samples.
        train_iters = args.train_iters
        eval_iters = (train_iters // args.eval_interval + 1) * args.eval_iters
        test_iters = args.eval_iters
        train_val_test_num_samples = [train_iters * global_batch_size,
                                      eval_iters * global_batch_size,
                                      test_iters * global_batch_size]
        print_rank_0(' > datasets target sizes (minimum size):')
        print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
        print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
        print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

        # Build the datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(
            train_val_test_num_samples)

        # Build dataloders.
        train_dataloader = make_data_loader(train_ds)
        valid_dataloader = make_data_loader(valid_ds)
        test_dataloader = make_data_loader(test_ds)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor(
            [int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(flags,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()

    # Shift the start iterations.
    args.iteration = 0
    if train_dataloader is not None:
        train_dataloader.batch_sampler.start_iter = args.iteration % \
            len(train_dataloader)
        print_rank_0('setting training data start iteration to {}'.
                     format(train_dataloader.batch_sampler.start_iter))
    if valid_dataloader is not None:
        start_iter_val = (args.iteration // args.eval_interval) * \
            args.eval_iters
        valid_dataloader.batch_sampler.start_iter = start_iter_val % \
            len(valid_dataloader)
        print_rank_0('setting validation data start iteration to {}'.
                     format(valid_dataloader.batch_sampler.start_iter))

    # Build iterators.
    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator