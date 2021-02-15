# Copyright Xiaozhi Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import numpy as np
import torch
from . import FairseqDataset, BaseWrapperDataset


class KeNegDataset(BaseWrapperDataset):

    def __init__(self, dataset ,args):
        super().__init__(dataset)
        self.ns=args.negative_sample_size

    def _map_indices(self, indices):
        tmp=[]
        for index in indices:
            tmp=tmp+list(range(index*self.ns,(index+1)*self.ns))
        return tmp

    def __getitem__(self, index):
        tmp=self._map_indices([index])
        return [self.dataset[x] for x in tmp]
    
    def collater(self,samples):
        return self.dataset.collater([y for x in samples for y in x])
    
    def __len__(self):
        return len(self.dataset)//self.ns
