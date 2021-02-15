# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
import numpy as np
from fairseq import utils
from sklearn.metrics import precision_score, f1_score, recall_score

from . import FairseqCriterion, register_criterion


@register_criterion('relation_extraction')
class RelationExtractionCriterion(FairseqCriterion):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--save-predictions', metavar='FILE',
                            help='file to save predictions to')
        parser.add_argument('--label-num', type=int, help='Total number of labels')
        parser.add_argument('--negative-label', type=int, help='The label of negative instances')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert hasattr(model, 'classification_heads') and \
            'tacred' in model.classification_heads, \
            "model must provide sentence classification head for --criterion=relation_extraction"

        reps, _ = model(
            **sample['net_input'],
            features_only=True,
            #classification_head_name='tacred',
        )
        #print(reps.size())
        reps = torch.gather(reps,1,sample['index'])
        #print(reps.size())
        reps_ = reps.view(-1,1,self.args.encoder_embed_dim*2)
        #print(reps_.size())
        assert reps.size(0)==reps_.size(0), "check the index"
        logits = model.classification_heads["tacred"](reps_)
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

        loss = F.nll_loss(
            F.log_softmax(logits, dim=-1, dtype=torch.float32),
            targets,
            reduction='sum',
        )

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
        }

        preds = logits.max(dim=1)[1].cpu().numpy()
        targets = targets.cpu().numpy()
        pid = np.where(targets!=self.args.negative_label)[0]
        labels=list(range(0,self.args.negative_label))+list(range(self.args.negative_label+1,self.args.num_classes))

        logging_output.update(
            ncorrect=np.sum((preds==targets)[pid]),
            npositive=len(pid),
            precision=precision_score(targets, preds, labels=labels, average='micro'),
            recall=recall_score(targets, preds, labels=labels, average='micro')
        )
        
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            npositive = sum(log.get('npositive', 0) for log in logging_outputs)
            precision = sum(log.get('precision', 0)*log.get('npositive', 0) for log in logging_outputs)/npositive
            recall = sum(log.get('recall', 0)*log.get('npositive', 0) for log in logging_outputs)/npositive
            
            agg_output.update(accuracy=ncorrect/npositive)
            agg_output.update(precision=precision)
            agg_output.update(recall=recall)
            agg_output.update(f1=2.0/(1.0/precision+1.0/recall))


        return agg_output
