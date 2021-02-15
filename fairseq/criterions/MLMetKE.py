# Copyright Xiaozhi Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('MLMetKE')
class MLMetKELoss(FairseqCriterion):
    """
    Implementation for the loss used in jointly training masked language model (MLM) and knowledge embedding (KE).
    """

    def __init__(self, args, task):
        super().__init__(args, task)
    def MLM_loss(self, model, sample):
        logits = model(**sample['MLM']['net_input'], return_all_hiddens=False)[0]
        targets = model.get_targets(sample["MLM"], [logits])
        loss = F.nll_loss(
            F.log_softmax(
                logits.view(-1, logits.size(-1)),
                dim=-1,
                dtype=torch.float32,
            ),
            targets.view(-1),
            reduction='mean',
            ignore_index=self.padding_idx,
        )

        sample_size = targets.ne(self.padding_idx).int().sum().item()
        return loss, sample_size

    def KE_loss(self, model, sample):
        relations = model.get_targets(sample["KE"],None)
        inputs=sample["KE"]["net_input"]
        pScores, nScores, sample_size = model.KEscore(src_tokens=(inputs["heads"],inputs["tails"],inputs["nHeads"],inputs["nTails"],inputs["heads_r"],inputs["tails_r"], inputs['relation_desc'] if 'relation_desc' in inputs else None),relations=relations,ke_head_name=self.args.ke_head_name) 
        pLoss = F.logsigmoid(pScores).squeeze(dim=1)
        nLoss = F.logsigmoid(-nScores).mean(dim=1)
        loss = (-pLoss.mean()-nLoss.mean())/2.0
        return loss, sample_size 

    def KE_loss2(self, model, sample):
        relations = model.get_targets(sample["KE2"],None)
        inputs=sample["KE2"]["net_input"]
        pScores, nScores, sample_size = model.KEscore(src_tokens=(inputs["heads"],inputs["tails"],inputs["nHeads"],inputs["nTails"],inputs["heads_r"],inputs["tails_r"], inputs['relation_desc'] if 'relation_desc' in inputs else None),relations=relations,ke_head_name=self.args.ke_head_name2) 
        pLoss = F.logsigmoid(pScores).squeeze(dim=1)
        nLoss = F.logsigmoid(-nScores).mean(dim=1)
        loss = (-pLoss.mean()-nLoss.mean())/2.0 
        return loss, sample_size 


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        mlm_loss, mlm_size = self.MLM_loss(model, sample)
        ke_loss, ke_size = self.KE_loss(model, sample)
        if 'KE2' in sample:
            ke2_loss, ke2_size = self.KE_loss2(model, sample)

        loss = mlm_loss + ke_loss
        if 'KE2' in sample:
            loss += ke2_loss
            ke_loss = (ke_loss + ke2_loss) / 2
        sample_size = 1
        #print("Sample size",sample_size,ke_size,mlm_size)
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample["MLM"]['ntokens']+sample["KE"]["ntokens"],
            'nsentences': sample["MLM"]['nsentences']+sample["KE"]["nsentences"],
            'sample_size': sample_size,
            'ke_loss' : utils.item(ke_loss.data) if reduce else ke_loss.data,
            'mlm_loss' : utils.item(mlm_loss.data) if reduce else mlm_loss.data,
        }
        return loss, sample_size, logging_output


    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        ke_loss = sum(log.get('ke_loss', 0) for log in logging_outputs)
        mlm_loss = sum(log.get('mlm_loss', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'ke_loss' : ke_loss / sample_size / math.log(2),
            'mlm_loss' : mlm_loss / sample_size / math.log(2),
        }

        return agg_output
