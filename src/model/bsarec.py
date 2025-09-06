import copy
import torch
import torch.nn as nn
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, FeedForward, MultiHeadAttention
from model.popularity import build_popularity_encoding, EvalPopularityEncoding

class BSARecModel(SequentialRecModel):
    def __init__(self, args):
        super(BSARecModel, self).__init__(args)
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = BSARecEncoder(args)
        self.use_popularity = getattr(args, "use_popularity", False)
        if self.use_popularity:
            self.popularity_enc = build_popularity_encoding(
                args.input_units1,
                args.input_units2,
                args.base_dim1,
                args.base_dim2,
                args.popularity_dir,
                args.data_name,
            )
            self.pop_embed = nn.Linear(
                args.input_units1 + args.input_units2, args.hidden_size
            )
            if getattr(args, "use_week_eval", False):
                self.eval_popularity_enc = build_popularity_encoding(
                    args.input_units1,
                    args.input_units2,
                    args.base_dim1,
                    args.base_dim2,
                    args.popularity_dir,
                    args.data_name,
                    enable_eval=True,
                    pause=args.pause,
                )
        else:
            self.popularity_enc = None
            self.pop_embed = None
        self.apply(self.init_weights)

    def forward(self, input_ids, time1_seq, time2_seq, user_ids=None, all_sequence_output=False):
        extended_attention_mask = self.get_attention_mask(input_ids)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        item_emb = self.item_embeddings(input_ids)
        pos_emb = self.position_embeddings(position_ids)
        sequence_emb = item_emb + pos_emb
        if self.use_popularity and self.popularity_enc is not None:
            if isinstance(self.popularity_enc, EvalPopularityEncoding):
                pop_feats = self.popularity_enc(input_ids, time1_seq, time2_seq, user_ids)
            else:
                pop_feats = self.popularity_enc(input_ids, time1_seq, time2_seq)
            pop_emb = self.pop_embed(pop_feats)
            sequence_emb = sequence_emb + pop_emb

        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        item_encoded_layers = self.item_encoder(
            sequence_emb,
            extended_attention_mask,
            output_all_encoded_layers=True,
        )
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output

    def calculate_loss(
        self,
        input_ids,
        answers,
        neg_answers,
        same_target,
        user_ids,
        time1_seq=None,
        time2_seq=None,
    ):
        seq_output = self.forward(input_ids, time1_seq, time2_seq, user_ids)
        seq_output = seq_output[:, -1, :]
        item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)

        return loss

    def predict(self, input_ids, user_ids=None, time1_seq=None, time2_seq=None, all_sequence_output=False):
        return self.forward(input_ids, time1_seq, time2_seq, user_ids, all_sequence_output)

class BSARecEncoder(nn.Module):
    def __init__(self, args):
        super(BSARecEncoder, self).__init__()
        self.args = args
        block = BSARecBlock(args)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = [ hidden_states ]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states) # hidden_states => torch.Size([256, 50, 64])
        return all_encoder_layers

class BSARecBlock(nn.Module):
    def __init__(self, args):
        super(BSARecBlock, self).__init__()
        self.layer = BSARecLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output

class BSARecLayer(nn.Module):
    def __init__(self, args):
        super(BSARecLayer, self).__init__()
        self.args = args
        self.filter_layer = FrequencyLayer(args)
        self.attention_layer = MultiHeadAttention(args)
        self.alpha = args.alpha

    def forward(self, input_tensor, attention_mask):
        dsp = self.filter_layer(input_tensor)
        gsp = self.attention_layer(input_tensor, attention_mask)
        hidden_states = self.alpha * dsp + ( 1 - self.alpha ) * gsp

        return hidden_states
    
class FrequencyLayer(nn.Module):
    def __init__(self, args):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.c = args.c // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, args.hidden_size))

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        low_pass = x[:]
        low_pass[:, self.c:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
