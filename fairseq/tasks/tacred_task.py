# Copyright (c) Xiaozhi Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

import numpy as np
import torch

from fairseq.data import (
    data_utils,
    Dictionary,
    BertDictionary,
    encoders,
    IdDataset,
    ListDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    RawLabelDataset,
    RightPadDataset,
    SortDataset,
)
from fairseq.tasks import FairseqTask, register_task
import transformers
from transformers import BertTokenizer


@register_task('tacred')
class TacredTask(FairseqTask):
    """Task to finetune RoBERTa for TACRED."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='DIR',
                            help='path to data directory; we load <split>.jsonl')
        parser.add_argument('--init-token', type=int, default=None,
                            help='add token at the beginning of each batch item')
        parser.add_argument('--num-classes', type=int, default=42)
        parser.add_argument('--regression-target', action='store_true', default=False)
    def __init__(self, args, vocab):
        super().__init__(args)
        self.vocab = vocab 
        if getattr(args, 'bert', False):
            self.mask = vocab.mask_index
            self.bpe = BertTokenizer.from_pretrained('bert-base-uncased')
            self.tokenizer = self.bpe
            print('| bert bpe')
        else:
            self.mask = vocab.add_symbol('<mask>')
            self.bpe = encoders.build_bpe(args)

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol('<mask>')
        #dictionary.add_symbol('<e1>')
        #dictionary.add_symbol('</e1>')
        #dictionary.add_symbol('<e2>')
        #dictionary.add_symbol('</e2>')
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        #assert args.criterion == 'relation_extraction', 'Must set --criterion=relation_extraction'

        # load data and label dictionaries
        if getattr(args, 'bert', False):
            print('| bert dictionary')
            vocab = BertDictionary()
        else:
            vocab = cls.load_dictionary(os.path.join(args.data, 'dict.txt'))
            print('| dictionary: {} types'.format(len(vocab)))

        return cls(args, vocab)

    def load_dataset(self, split, epoch=0, combine=False, data_path=None, return_only=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        def getIns(bped,bpeTokens,tokens,L,R):
            resL=0
            tkL=" ".join(tokens[:L])
            bped_tkL=self.bpe.encode(tkL)
            if bped.find(bped_tkL)==0:
                resL=len(bped_tkL.split())
            else:
                tkL+=" "
                bped_tkL=self.bpe.encode(tkL)
                if bped.find(bped_tkL)==0:
                    resL=len(bped_tkL.split())
            resR=0
            tkR=" ".join(tokens[R:])
            bped_tkR=self.bpe.encode(tkR)
            if bped.rfind(bped_tkR)+len(bped_tkR)==len(bped):
                resR=len(bpeTokens)-len(bped_tkR.split())
            else:
                tkR=" "+tkR
                bped_tkR=self.bpe.encode(tkR)
                if bped.rfind(bped_tkR)+len(bped_tkR)==len(bped):
                    resR=len(bpeTokens)-len(bped_tkR.split())
            return resL, resR
        
        def getExample(a,bias):
            s=" ".join(a["token"])
            ss=self.bpe.encode(s)
            sst=ss.split()
            headL=a['h']['pos'][0]
            headR=a['h']['pos'][1]
            hiL, hiR=getIns(ss,sst,a["token"],headL,headR)
            tailL=a['t']['pos'][0]
            tailR=a['t']['pos'][1]
            tiL, tiR=getIns(ss,sst,a["token"],tailL,tailR)
            E1b='1'
            E1e='2'
            E2b='3'
            E2e='4'
            ins=[(hiL, E1b), (hiR, E1e), (tiL, E2b), (tiR, E2e)]
            ins=sorted(ins)
            pE1=0
            pE2=0
            pE1_=0
            pE2_=0
            for i in range(0,4):
                sst.insert(ins[i][0]+i,ins[i][1])
                if ins[i][1]==E1b:
                    pE1=ins[i][0]+i
                elif ins[i][1]==E2b:
                    pE2=ins[i][0]+i
                elif ins[i][1]==E1e:
                    pE1_=ins[i][0]+i
                else:
                    pE2_=ins[i][0]+i
            if pE1_-pE1==1 or pE2_-pE2==1:
                return "???", -1, -1
            else:
                return " ".join(sst), pE1+bias, pE2+bias

        def get_example_bert(item):
            if 'text' in item:
                sentence = item['text']
                is_token = False
            else:
                sentence = item['token']
                is_token = True
            pos_head = item['h']['pos']
            pos_tail = item['t']['pos']

            pos_min = pos_head
            pos_max = pos_tail
            if pos_head[0] > pos_tail[0]:
                pos_min = pos_tail
                pos_max = pos_head
                rev = True
            else:
                rev = False
            
            if not is_token:
                sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
                ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
                sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
                ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
                sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
            else:
                sent0 = self.tokenizer.tokenize(' '.join(sentence[:pos_min[0]]))
                ent0 = self.tokenizer.tokenize(' '.join(sentence[pos_min[0]:pos_min[1]]))
                sent1 = self.tokenizer.tokenize(' '.join(sentence[pos_min[1]:pos_max[0]]))
                ent1 = self.tokenizer.tokenize(' '.join(sentence[pos_max[0]:pos_max[1]]))
                sent2 = self.tokenizer.tokenize(' '.join(sentence[pos_max[1]:]))

            ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
            ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']

            re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
            pos1 = 1 + len(sent0) if not rev else 1 + len(sent0 + ent0 + sent1)
            pos2 = 1 + len(sent0 + ent0 + sent1) if not rev else 1 + len(sent0)
            #pos1 = min(self.max_length - 1, pos1)
            #pos2 = min(self.max_length - 1, pos2)
            
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
            avai_len = len(indexed_tokens)

            # Position
            #pos1 = torch.tensor([[pos1]]).long()
            #pos2 = torch.tensor([[pos2]]).long()

            #indexed_tokens = indexed_tokens[:self.max_length]
            indexed_tokens = torch.tensor(indexed_tokens).long()

            return indexed_tokens, pos1, pos2

 
        def binarize(s, append_bos=False):
            #if self.bpe is not None:
            #    s = self.bpe.encode(s)
            tokens = self.vocab.encode_line(
                s, append_eos=True, add_if_not_exist=False,
            ).long()
            if append_bos and self.args.init_token is not None:
                tokens = torch.cat([tokens.new([self.args.init_token]), tokens])
            return tokens

        if data_path is None:
            data_path = os.path.join(self.args.data, split + '.jsonl')
            rel2id_path=os.path.join(self.args.data, "rel2id.json")
        if not os.path.exists(data_path):
            raise FileNotFoundError('Cannot find data: {}'.format(data_path))
        if not os.path.exists(rel2id_path):
            raise FileNotFoundError('Cannot find rel2id: {}'.format(rel2id_path))
        
        rel2id=json.load(open(rel2id_path,"r"))
        labels = []
        src_tokens = []
        src_lengths = []
        src_idx = []
        with open(data_path) as h:
            for line in h:
                example = json.loads(line.strip())
                if 'relation' in example:
                    label = rel2id[example['relation']]
                    labels.append(label)
                #bped=self.bpe.encode(" ".join(example["token"]))
                if getattr(self.args, 'bert', False):
                    src_bin, pE1, pE2 = get_example_bert(example)
                else:
                    bped, pE1, pE2 = getExample(example,1)
                    if pE1==-1:
                        continue
                    src_bin = binarize(bped, append_bos=True)
                src_tokens.append(src_bin)
                src_lengths.append(len(src_bin))
                #pE1=0
                #pE2=0
                src_idx.append([[pE1 for i in range(0,self.args.encoder_embed_dim)], [pE2 for i in range(0,self.args.encoder_embed_dim)]])

        src_lengths = np.array(src_lengths)
        src_tokens = ListDataset(src_tokens, src_lengths)
        src_lengths = ListDataset(src_lengths)
        
        print("src_len", len(src_lengths))
        print("src_tokens", len(src_tokens))
        

        dataset = {
            'id': IdDataset(),
            'net_input':{
                'src_tokens':RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad()
                ),
                'src_lengths': src_lengths,
            },
            'index': RawLabelDataset(src_idx),
            'target': RawLabelDataset(labels),
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(src_tokens, reduce=True),
        }
        
        
        dataset = NestedDictionaryDataset(
            dataset,
            sizes=src_tokens.sizes,
        )

        with data_utils.numpy_seed(self.args.seed+epoch):
            dataset = SortDataset(
                dataset,
                # shuffle
                sort_order=[np.random.permutation(len(dataset))],
            )

        print('| Loaded {} with {} samples'.format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        
        model.register_classification_head(
            'tacred',
            num_classes=args.num_classes,
            inner_dim=2*args.encoder_embed_dim,
            input_dim=2*args.encoder_embed_dim
        )
        '''
        model.register_classification_head(
            'sentence_classification_head',
            num_classes=args.num_classes,
        )
        '''
        return model

    @property
    def source_dictionary(self):
        return self.vocab

    @property
    def target_dictionary(self):
        return self.vocab
