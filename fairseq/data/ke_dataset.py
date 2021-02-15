# Copyright Xiaozhi Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import numpy as np
import torch
from . import FairseqDataset


class KEDataset(FairseqDataset):

    def __init__(self, head, tail, nHead, nTail, count, sizes,args):
        super().__init__()
        self.head=head
        self.tail=tail
        self.nHead=nHead
        self.nTail=nTail
        self.count=count
        self.sizes=sizes
        self.ns=args.negative_sample_size
    def _map_indices(self, indices):
        tmp=[]
        for index in indices:
            tmp=tmp+list(range(index*self.ns,(index+1)*self.ns))
        return tmp

    def __getitem__(self, index):
        head=self.head[index]
        tail=self.tail[index]
        tmp=self._map_indices([index])
        nHead=[self.nHead[x] for x in tmp]
        nTail=[self.nTail[x] for x in tmp]
        return head, tail, nHead, nTail

    def __len__(self):
        return len(self.head)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        heads=self.head.collater([x[0] for x in samples])
        tails=self.tail.collater([x[1] for x in samples])
        nHeads=self.nHead.collater([y for x in samples for y in x[2]])
        nTails=self.nTail.collater([y for x in samples for y in x[3]])
        return heads, tails, nHeads, nTails

    def num_tokens(self, index):
        """Return an example's length (number of tokens), used for batching."""
        # TODO make it configurable whether to use max() or sum() here
        tmp=self._map_indices([index])
        a=sum([self.nHead.num_tokens(x) for x in tmp])
        b=sum([self.nTail.num_tokens(x) for x in tmp])
        return max(max(a,b),max(self.head.num_tokens(index),self.tail.num_tokens(index)))

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return 500

    @property
    def supports_prefetch(self):
        return self.head.supports_prefetch and self.tail.supports_prefetch and self.nHead.supports_prefetch and self.nTail.supports_prefetch
    
    def prefetch(self, indices):
        self.head.prefetch(indices)
        self.tail.prefetch(indices)
        tmp=self._map_indices(indices)
        self.nHead.prefetch(tmp)
        self.nTail.prefetch(tmp)
