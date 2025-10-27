import copy
import torch
import torch.nn as nn
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, FeedForward, MultiHeadAttention


class BSARecModel(SequentialRecModel):
    def __init__(self, args):
        super(BSARecModel, self).__init__(args)
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = BSARecEncoder(args)
        self.use_popularity = getattr(args, "use_popularity", False)
        self.use_long_popularity = getattr(args, "use_long_popularity", True)
        self.use_short_popularity = getattr(args, "use_short_popularity", True)
        if self.use_popularity:
            self.pop_dropout = nn.Dropout(args.hidden_dropout_prob)
            if self.use_long_popularity:
                self.pop_long_mapper = nn.Sequential(
                    nn.Linear(args.hidden_size + args.pop_long_dim, args.hidden_size),
                    nn.GELU(),
                )
                self.pop_long_norm = LayerNorm(args.hidden_size, eps=1e-12)
            if self.use_short_popularity:
                self.pop_short_mapper = nn.Sequential(
                    nn.Linear(args.hidden_size + args.pop_short_dim, args.hidden_size),
                    nn.GELU(),
                )
                self.pop_short_norm = LayerNorm(args.hidden_size, eps=1e-12)
        self.apply(self.init_weights)

    def forward(self, input_ids, pop_long=None, pop_short=None, user_ids=None, all_sequence_output=False):
        extended_attention_mask = self.get_attention_mask(input_ids)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        item_emb = self.item_embeddings(input_ids)
        pos_emb = self.position_embeddings(position_ids)
        if self.use_popularity and self.use_long_popularity and pop_long is not None:
            long_input = torch.cat([item_emb, pop_long], dim=-1)
            long_sequence_emb = self.pop_long_mapper(long_input)
            long_sequence_emb = self.pop_long_norm(long_sequence_emb)
            long_sequence_emb = self.pop_dropout(long_sequence_emb)
        else:
            long_sequence_emb = item_emb

        if self.use_popularity and self.use_short_popularity and pop_short is not None:
            short_input = torch.cat([item_emb, pop_short], dim=-1)
            short_sequence_emb = self.pop_short_mapper(short_input)
            short_sequence_emb = self.pop_short_norm(short_sequence_emb)
            short_sequence_emb = self.pop_dropout(short_sequence_emb)
        else:
            short_sequence_emb = item_emb

        long_sequence_emb = long_sequence_emb + pos_emb
        short_sequence_emb = short_sequence_emb + pos_emb

        item_encoded_layers = self.item_encoder(
            long_sequence_emb,
            short_sequence_emb,
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
            pop_long=None,
            pop_short=None,
    ):
        seq_output = self.forward(input_ids, pop_long, pop_short, user_ids)
        seq_output = seq_output[:, -1, :]
        item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)

        return loss

    def predict(self, input_ids, user_ids=None, pop_long=None, pop_short=None, all_sequence_output=False):
        return self.forward(input_ids, pop_long, pop_short, user_ids, all_sequence_output)


class BSARecEncoder(nn.Module):
    def __init__(self, args):
        super(BSARecEncoder, self).__init__()
        self.args = args
        block = BSARecBlock(args)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, long_hidden_states, short_hidden_states, attention_mask, output_all_encoded_layers=False):
        hidden_states_long = long_hidden_states
        hidden_states_short = short_hidden_states
        all_encoder_layers = [hidden_states_long]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states_long, hidden_states_short, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
            hidden_states_long = hidden_states
            hidden_states_short = hidden_states
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)  # hidden_states => torch.Size([256, 50, 64])
        return all_encoder_layers


class BSARecBlock(nn.Module):
    def __init__(self, args):
        super(BSARecBlock, self).__init__()
        self.layer = BSARecLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, long_hidden_states, short_hidden_states, attention_mask):
        layer_output = self.layer(long_hidden_states, short_hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output


class BSARecLayer(nn.Module):
    def __init__(self, args):
        super(BSARecLayer, self).__init__()
        self.args = args
        self.use_frequency = getattr(args, "use_frequency_domain", True)
        self.use_time = getattr(args, "use_time_domain", True)
        if self.use_frequency:
            self.filter_layer = FrequencyLayer(args)
        if self.use_time:
            self.attention_layer = MultiHeadAttention(args)
        hidden = args.hidden_size
        # Learnable gating network for adaptive fusion of frequency and feature domains
        if self.use_frequency and self.use_time:
            self.gate = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.Sigmoid(),
            )

    def forward(self, long_input_tensor, short_input_tensor, attention_mask):
        dsp = None
        gsp = None
        if self.use_frequency:
            dsp = self.filter_layer(short_input_tensor)
        if self.use_time:
            gsp = self.attention_layer(long_input_tensor, attention_mask)

        if self.use_frequency and self.use_time:
            fusion = torch.cat([dsp, gsp], dim=-1)
            gate = self.gate(fusion)
            hidden_states = gate * dsp + (1 - gate) * gsp
        elif self.use_frequency:
            hidden_states = dsp
        elif self.use_time:
            hidden_states = gsp
        else:
            hidden_states = long_input_tensor
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
        sequence_emb_fft = low_pass + (self.sqrt_beta ** 2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states