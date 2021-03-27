import torch
import torch.nn.functional as F

from summa import get_args
from summa import print_rank_0
from summa import get_timers
from summa import mpu
from summa.data.dataset_utils import build_train_valid_test_datasets
# from megatron.model import BertModel
# from summa.model import BertModel
# from summa.utils import reduce_losses


# def model_provider():
#     """Build the model."""
#
#     print_rank_0('building BERT model ...')
#
#     model = BertModel(
#         num_tokentypes=2,
#         add_binary_head=True,
#         parallel_output=True)
#
#     return model


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


# def forward_step(data_iterator, model):
#     """Forward step."""
#     args = get_args()
#     timers = get_timers()
#
#     # Get the batch.
#     timers('batch generator').start()
#     tokens, types, sentence_order, loss_mask, lm_labels, padding_mask \
#         = get_batch(data_iterator)
#     timers('batch generator').stop()
#
#     # Forward model. lm_labels
#     lm_loss_, sop_logits = model(tokens, padding_mask,
#                                  tokentype_ids=types,
#                                  lm_labels=lm_labels)
#     sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(),
#                                sentence_order.view(-1),
#                                ignore_index=-1)
#
#     lm_loss = torch.sum(
#         lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()
#
#     loss = lm_loss + sop_loss
#
#     reduced_losses = reduce_losses([lm_loss, sop_loss])
#
#     return loss, {'lm loss': reduced_losses[0], 'sop loss': reduced_losses[1]}


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for BERT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        max_seq_length=args.seq_length,
        masked_lm_prob=args.mask_prob,
        short_seq_prob=args.short_seq_prob,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating BERT datasets ...")

    return train_ds, valid_ds, test_ds


from develop import Develop

if __name__ == "__main__":

    # Develop(train_valid_test_datasets_provider, model_provider,
    #         args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
    Develop(train_valid_test_datasets_provider, None,
            args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})