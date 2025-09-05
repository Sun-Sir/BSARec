import torch
import torch.nn as nn

class PopularityEncoding(nn.Module):
    """Simple popularity-based encoding.

    This module produces item popularity features for a sequence of
    item indices. It concatenates two trainable embeddings that
    correspond to different temporal resolutions. The time sequences
    are accepted for interface compatibility but are not used in the
    simplified implementation.
    """

    def __init__(self, args):
        super().__init__()
        self.item_pop1 = nn.Embedding(args.item_size, args.input_units1)
        self.item_pop2 = nn.Embedding(args.item_size, args.input_units2)

    def forward(self, log_seqs, time1_seqs, time2_seqs):
        pop1 = self.item_pop1(log_seqs)
        pop2 = self.item_pop2(log_seqs)
        return torch.cat([pop1, pop2], dim=-1)
