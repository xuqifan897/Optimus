import torch
from summa import get_args
from summa.model.utils import init_method_normal, scaled_init_method_normal
from summa.module import OptimusModule
from summa.model_new.transformer import ParallelTransformer
import summa.mpu as mpu
from .BertLMHead import checkpoint_in_conjunction
from summa.mpu.embeddings import Vocab_Position_Tokentype_ParallelEmbedding

def get_language_model(attention_mask_func, num_tokentypes, add_pooler,
                       init_method=None, scaled_init_method=None):
    args = get_args()

    if init_method is None:
        init_method = init_method_normal(args.init_method_std)

    if scaled_init_method is None:
        scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)

    # Language model.
    language_model = TransformerLanguageModel(
        attention_mask_func=attention_mask_func,
        init_method=init_method,
        output_layer_init_method=scaled_init_method,
        num_tokentypes=num_tokentypes,
        add_pooler=add_pooler)
    # key used for checkpoints.
    language_model_key = 'language_model'

    return language_model, language_model_key


class Pooler(OptimusModule):
    """Pooler layer.

        Pool hidden states of a specific token (for example start of the
        sequence) and add a linear transformation followed by a tanh.

        Arguments:
            hidden_size: hidden size
            init_method: weight initialization method for the linear layer.
                bias is set to zero.
        """
    def __init__(self, hidden_size, init_method):
        super(Pooler, self).__init__()
        self.dense = mpu.SUMMALinear(
            hidden_size, hidden_size,
            bias_flag=True, init_method=init_method,
            parameter_gradient_buffer=mpu.get_parameter_gradient_buffer())

    def forward(self, hidden_states, sequence_idx):
        # hidden_states: [b/q, s, h/q]
        # sequence index: index of the token to pool
        pooled = hidden_states[:, sequence_idx, :]
        pooled = pooled.unsqueeze(1)
        pooled = self.dense(pooled)
        pooled = torch.tanh(pooled)
        return pooled


class TransformerLanguageModel(OptimusModule):
    def __init__(self,
                 attention_mask_func,
                 init_method,
                 output_layer_init_method,
                 num_tokentypes=0,
                 add_pooler=False):
        super(TransformerLanguageModel, self).__init__()
        args = get_args()

        self.hidden_size = args.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method = init_method
        self.add_pooler = add_pooler

        self.embedding = Vocab_Position_Tokentype_ParallelEmbedding(
            args.padded_vocab_size, self.hidden_size, num_tokentypes, init_method)
        self._embedding_key = 'embedding'

        # Transformer
        self.transformer = ParallelTransformer(
            attention_mask_func, self.init_method,
            output_layer_init_method)
        self._transformer_key = 'transformer'

        if self.add_pooler:
            self.pooler = Pooler(self.hidden_size, self.init_method)
            self._pooler_key = 'pooler'

    def forward(self, input_ids, position_ids, attention_mask,
                tokentype_ids=None, pooling_sequence_index=0):

        # Embeddings.
        embedding_output = self.embedding(input_ids, tokentype_ids)

        # Transformer.
        transformer_output = self.transformer(
            embedding_output, attention_mask)
        # transformer_output = checkpoint_in_conjunction.apply(transformer_output)

        if self.add_pooler:
            pooled_output = self.pooler(
                transformer_output, pooling_sequence_index)
            return transformer_output, pooled_output
        else:
            return transformer_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._embedding_key] \
            = self.embedding.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars)
        state_dict_[self._transformer_key] \
            = self.transformer.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars)
        if self.add_pooler:
            state_dict_[self._pooler_key] \
                = self.pooler.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Embedding.
        if self._embedding_key in state_dict:
            state_dict_ = state_dict[self._embedding_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if '_embeddings' in key:
                    state_dict_[key] = state_dict[key]
        self.embedding.load_state_dict(state_dict_, strict=strict)

        # Transformer.
        if self._transformer_key in state_dict:
            state_dict_ = state_dict[self._transformer_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'transformer.' in key:
                    state_dict_[key.split('transformer.')[1]] = state_dict[key]
        self.transformer.load_state_dict(state_dict_, strict=strict)

        # Pooler.
        if self.add_pooler:
            assert 'pooler' in state_dict, \
                'could not find data for pooler in the checkpoint'
            self.pooler.load_state_dict(state_dict[self._pooler_key],
                                        strict=strict)