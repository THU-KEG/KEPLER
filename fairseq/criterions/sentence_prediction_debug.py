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


@register_criterion('sentence_prediction_debug')
class SentencePredictionDebugCriterion(FairseqCriterion):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--save-predictions', metavar='FILE',
                            help='file to save predictions to')
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
            'sentence_classification_head' in model.classification_heads, \
            "model must provide sentence classification head for --criterion=sentence_prediction"

        logits, _ = model(
            **sample['net_input'],
            features_only=True,
            classification_head_name='sentence_classification_head',
        )
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

        if not self.args.regression_target:
            loss = F.nll_loss(
                F.log_softmax(logits, dim=-1, dtype=torch.float32),
                targets,
                reduction='sum',
                #ignore_index=self.args.negative_label,
            )
        else:
            logits = logits.squeeze().float()
            targets = targets.float()
            loss = F.mse_loss(
                logits,
                targets,
                reduction='sum',
            )

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
        }

        if not self.args.regression_target:
            #preds = logits.max(dim=1)[1]
            preds = logits.max(dim=1)[1].cpu().numpy()
            targets = targets.cpu().numpy()
            pid = np.where(targets!=self.args.negative_label)[0]
            labels=list(range(0,self.args.negative_label))+list(range(self.args.negative_label+1,self.args.num_classes))
            #print(preds)
            #print(targets)
            #print(preds==targets)
            #print(pid)
            logging_output.update(
                ncorrect=np.sum((preds==targets)[pid]),
                nsentences=len(pid),
                #sample_size=len(pid),
            )
            '''
            if len(pid)==0:
                print(preds)
                print(targets)
                print(preds==targets)
                print(pid)
                print("Cor",logging_output["ncorrect"])
                print("Poi",logging_output["nsentences"])
            '''
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
        #print("ncorrect",ncorrect) 
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'ncorrect': ncorrect,
        }

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            accuracy=ncorrect/nsentences
            agg_output.update(accuracy=accuracy if nsentences!=0 else 0.0)

        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
