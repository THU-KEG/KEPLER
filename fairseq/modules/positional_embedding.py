# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from .learned_positional_embedding import LearnedPositionalEmbedding
from .sinusoidal_positional_embedding import SinusoidalPositionalEmbedding

class BertPositionalEmbedding(nn.Embedding):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.onnx_trace = False

    def forward(self, input, incremental_state=None, positions=None):
        """Input is expected to be of size [bsz x seqlen]."""
        return super().forward(positions)

def PositionalEmbedding(
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        learned: bool = False,
        bert_pos_emb: bool = False,
):
    if bert_pos_emb:
        return BertPositionalEmbedding(num_embeddings, embedding_dim)
    elif learned:
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately
        # TODO: The right place for this offset would be inside
        # LearnedPositionalEmbedding. Move this there for a cleaner implementation.
        if padding_idx is not None:
            num_embeddings = num_embeddings + padding_idx + 1
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        if padding_idx is not None:
            nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(
            embedding_dim, padding_idx, init_size=num_embeddings + padding_idx + 1,
        )
    return m
