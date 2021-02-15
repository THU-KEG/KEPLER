# Copyright Xiaozhi Wang.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import json

from fairseq.data import (
    ConcatDataset,
    ConcatSentencesDataset,
    data_utils,
    Dictionary,
    BertDictionary,
    encoders,
    IdDataset,
    indexed_dataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PadDataset,
    RightPadDataset,
    PrependTokenDataset,
    SortDataset,
    TokenBlockDataset,
    FakeNumelDataset,
    TruncateDataset,
    KEDataset,
    RawLabelDataset,
    RoundRobinZipDatasets,
    KeNegDataset,
)
from fairseq.tasks import FairseqTask, register_task


@register_task('MLMetKE')
class MLMetKETask(FairseqTask):
    """Task for jointly training masked language models and Knowledge Embedding."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='colon separated path to data directories list, \
        will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--KEdata', help='file prefix for knowledge embedding data')
        parser.add_argument('--KEdata2', help='file prefix for the second knowledge embedding data', default='')
        parser.add_argument('--sample-break-mode', default='complete', choices=['none', 'complete', 'complete_doc', 'eos'], help='If omitted or "none", fills each sample with tokens-per-sample '
                                 'tokens. If set to "complete", splits samples only at the end '
                                 'of sentence, but may include multiple sentences per sample. '
                                 '"complete_doc" is similar but respects doc boundaries. '
                                 'If set to "eos", includes only one sentence per sample.')
        parser.add_argument('--tokens-per-sample', default=512, type=int,
                            help='max number of total tokens over all segments '
                                 'per sample for BERT dataset')
        parser.add_argument('--mask-prob', default=0.15, type=float,
                            help='probability of replacing a token with mask')
        parser.add_argument('--leave-unmasked-prob', default=0.1, type=float,
                            help='probability that a masked token is unmasked')
        parser.add_argument('--random-token-prob', default=0.1, type=float,
                            help='probability of replacing a token with a random token')
        parser.add_argument('--freq-weighted-replacement', action='store_true',
                            help='sample random replacement words based on word frequencies')
        parser.add_argument('--mask-whole-words', default=False, action='store_true',
                            help='mask whole words; you may also want to set --bpe')
        parser.add_argument('--negative-sample-size', default=1, type=int,
                            help='The number of negative samples per positive sample for Knowledge Embedding' )
        parser.add_argument('--ke-model', default='TransE', type=str,
                            help='Knowledge Embedding Method (TransE, RotatE, etc)')
        parser.add_argument('--ke-head-name', default='wikiData', type=str,
                            help='Knowledge Embedding head name (wikiData , etc)')
        parser.add_argument('--ke-head-name2', default='wordnet', type=str,
                            help='Knowledge Embedding head name (wikiData , etc)')
        parser.add_argument('--init-token', type=int, default=None,
                            help='add token at the beginning of each batch item')
        parser.add_argument('--separator-token', type=int, default=None,
                            help='add separator token between inputs')
        parser.add_argument('--gamma', type=float, default=12.0)
        parser.add_argument('--gamma2', type=float, default=12.0)
        parser.add_argument('--nrelation', type=int, default=822)
        parser.add_argument('--nrelation2', type=int, default=20)
        parser.add_argument('--relation_desc', action='store_true')
        parser.add_argument('--double_ke', action='store_true')
        parser.add_argument('--relemb_from_desc', action='store_true')

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # add mask token
        if 'bert' in args and args.bert:
            self.mask_idx = dictionary.mask_index
        else:
            self.mask_idx = dictionary.add_symbol('<mask>')

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = args.data.split(':')
        assert len(paths) > 0
        if 'bert' in args and args.bert:
            print('| bert dictionary')
            dictionary = BertDictionary()
        else:
            dictionary = Dictionary.load(os.path.join(paths[0],'dict.txt'))
        print('| dictionary: {} types'.format(len(dictionary)))
        if args.freq_weighted_replacement:
            print('| freq weighted mask replacement')
        return cls(args, dictionary)

    def load_MLM_dataset(self, split, epoch=0, combine=False):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.source_dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample - 1,  # one less for <s>
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode=self.args.sample_break_mode,
        )

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

        # create masked input and targets
        
        if self.args.mask_whole_words:
            print('| mask whole words')
            bpe = encoders.build_bpe(self.args)
            if bpe is not None:

                def is_beginning_of_word(i):
                    if i < self.source_dictionary.nspecial:
                        # special elements are always considered beginnings
                        return True
                    tok = self.source_dictionary[i]
                    if tok.startswith('madeupword'):
                        return True
                    try:
                        return bpe.is_beginning_of_word(tok)
                    except ValueError:
                        return True

                Mask_whole_words = torch.ByteTensor(list(
                    map(is_beginning_of_word, range(len(self.source_dictionary)))
                ))
        else:
            print('| NO mask whold words')
            Mask_whole_words = None
        
        src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
            dataset,
            self.source_dictionary,
            pad_idx=self.source_dictionary.pad(),
            mask_idx=self.mask_idx,
            seed=self.args.seed,
            mask_prob=self.args.mask_prob,
            leave_unmasked_prob=self.args.leave_unmasked_prob,
            random_token_prob=self.args.random_token_prob,
            freq_weighted_replacement=self.args.freq_weighted_replacement,
            mask_whole_words=Mask_whole_words,
        )

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(src_dataset))

        dataset=SortDataset(
            NestedDictionaryDataset(
                {
                    'id': IdDataset(),
                    'net_input': {
                        'src_tokens': PadDataset(
                            src_dataset,
                            pad_idx=self.source_dictionary.pad(),
                            left_pad=False,
                        ),
                        'src_lengths': NumelDataset(src_dataset, reduce=False),
                    },
                    'target': PadDataset(
                        tgt_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    'nsentences': NumSamplesDataset(),
                    'ntokens': NumelDataset(src_dataset, reduce=True),
                },
                sizes=[src_dataset.sizes],
            ),
            sort_order=[
                shuffle,
                src_dataset.sizes
            ],
        )
        return dataset

    def load_KE_dataset(self, split, kedata_path, epoch=0, combine=False):
        paths = kedata_path.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]
        def get_path(type):
            return os.path.join(data_path,type,split)
        def desc_dataset(type, dictionary, relation_desc=None):
            now_path=get_path(type)
            #print(now_path)
            dataset=data_utils.load_indexed_dataset(
                now_path,
                dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            if self.args.init_token is not None:
                dataset = PrependTokenDataset(dataset, self.args.init_token)
            if relation_desc is not None:
                dataset = ConcatSentencesDataset(dataset, relation_desc)
            dataset = TruncateDataset(dataset, self.args.tokens_per_sample) #???
            dataset = RightPadDataset(dataset, pad_idx=self.source_dictionary.pad())
            return dataset
        
        assert(not (self.args.relation_desc and self.args.relemb_from_desc))

        if self.args.relation_desc or self.args.relemb_from_desc:
            now_path=get_path('relation_desc')
            relation_desc=data_utils.load_indexed_dataset(
                now_path,
                self.source_dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            if self.args.relation_desc:
                if self.args.separator_token is not None:
                    relation_desc = PrependTokenDataset(relation_desc, self.args.separator_token)
                else:
                    raise Exception("separator_token is None")
            elif self.args.relemb_from_desc:
                relation_desc = PrependTokenDataset(relation_desc, self.args.init_token)
                relation_desc = TruncateDataset(relation_desc, self.args.tokens_per_sample // 8) # 64
                relation_desc = RightPadDataset(relation_desc, pad_idx=self.source_dictionary.pad())
        else:
            relation_desc = None

        head=desc_dataset("head",self.source_dictionary)
        tail=desc_dataset("tail",self.source_dictionary)
        nHead=desc_dataset("negHead",self.source_dictionary)
        nTail=desc_dataset("negTail",self.source_dictionary)

        head_r=desc_dataset("head",self.source_dictionary, relation_desc if self.args.relation_desc else None)
        tail_r=desc_dataset("tail",self.source_dictionary, relation_desc if self.args.relation_desc else None)
        
        assert len(nHead)%len(head)==0, "check the KE positive and negative instances' number"
        self.negative_sample_size=len(nHead)/len(head)

        relation=np.load(get_path("relation")+".npy")
        sizes=np.load(get_path("sizes")+".npy")
        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle=np.random.permutation(len(head))
        net_input = {
            'heads': head,
            'tails': tail,
            'nHeads': KeNegDataset(nHead,self.args),
            'nTails': KeNegDataset(nTail,self.args),
            'heads_r': head_r,
            'tails_r': tail_r,
            'src_lengths': FakeNumelDataset(sizes, reduce=False),
        }
        if self.args.relemb_from_desc:
            net_input['relation_desc'] = relation_desc

        dataset=SortDataset(
            NestedDictionaryDataset(
                {
                    'id':IdDataset(),
                    'net_input': net_input,
                    'target': RawLabelDataset(relation),
                    'nsentences':NumSamplesDataset(),
                    'ntokens': FakeNumelDataset(sizes, reduce=True),
                },
                sizes=[sizes],
            ),
            sort_order=[shuffle],
        )
        return dataset

    def load_dataset(self, split, epoch=0, combine=False):
        MLMdataset=self.load_MLM_dataset(split,epoch,combine)
        # First KE data
        KEdataset=self.load_KE_dataset(split,self.args.KEdata, epoch,combine)
        # Second KE data
        if self.args.double_ke:
            KEdataset2=self.load_KE_dataset(split,self.args.KEdata2, epoch,combine)
            print("MLMdata",len(MLMdataset),"KEdata",len(KEdataset), "KEdata2", len(KEdataset2))
        else:
            print("MLMdata",len(MLMdataset),"KEdata",len(KEdataset))

        if self.args.double_ke:
            self.datasets[split]=RoundRobinZipDatasets(
                OrderedDict([("MLM",MLMdataset),
                            ("KE",KEdataset),
                            ("KE2",KEdataset2)
                ]),
                eval_key=None,
            )
        else:
            self.datasets[split]=RoundRobinZipDatasets(
                OrderedDict([("MLM",MLMdataset),
                            ("KE",KEdataset)
                ]),
                eval_key=None,
            )

        return self.datasets[split]

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        src_dataset = PadDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                self.args.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode='eos',
            ),
            pad_idx=self.source_dictionary.pad(),
            left_pad=False,
        )
        src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
        src_dataset = NestedDictionaryDataset(
            {
                'id': IdDataset(),
                'net_input': {
                    'src_tokens': src_dataset,
                    'src_lengths': NumelDataset(src_dataset, reduce=False),
                },
            },
            sizes=src_lengths,
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset
    
    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        model.register_ke_head(
            args.ke_head_name,
            gamma=args.gamma,
            nrelations=args.nrelation
        )
        if self.args.double_ke:
            model.register_ke_head(
                args.ke_head_name2,
                gamma=args.gamma2,
                nrelations=args.nrelation2
            )

        return model
    
    def max_positions(self):
        return (512,2147483647)

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
    
    def get_average_masked_score(self, model, src_tokens, mask, **net_input):
        """Mask a set of tokens and return their average score."""
        masked_tokens = src_tokens.clone()
        masked_tokens[mask.byte()] = self.mask_idx
        net_output = model(src_tokens=masked_tokens, **net_input, last_state_only=True)
        lprobs = F.log_softmax(net_output[0], dim=-1, dtype=torch.float32)
        lprobs = lprobs.gather(-1, src_tokens.unsqueeze(-1)).squeeze(-1)
        mask = mask.type_as(lprobs)
        score = (lprobs * mask).sum(dim=-1) / mask.sum(dim=-1)
        return score
