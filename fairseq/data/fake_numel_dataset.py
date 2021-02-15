# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from . import FairseqDataset


class FakeNumelDataset(FairseqDataset):

    def __init__(self, cnt, reduce=False):
        super().__init__()
        self.cnt = cnt
        self.reduce = reduce

    def __getitem__(self, index):
        return self.cnt[index]

    def __len__(self):
        return len(self.cnt)

    def collater(self, samples):
        if self.reduce:
            return sum(samples)
        else:
            #print(samples)
            #print("________________")
            return torch.tensor(samples)
