import numpy as np
import torch
import torch.nn.functional as F
import json
import os
import argparse
from fairseq.models.roberta import RobertaModel
from fairseq.data import (
    ConcatDataset,
    data_utils,
    Dictionary,
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
parser=argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="path to the entity decription data")
parser.add_argument("--ckpt_dir", type=str, help="path of the checkpoint")
parser.add_argument("--ckpt", type=str, help="filename of the checkpoint")
parser.add_argument("--dict", type=str, help="path to the dict.txt file", default="bpe/dict.txt")
parser.add_argument("--ent_emb", type=str, default="EntityEmb.npy", help="filename to save entity embedding")
parser.add_argument("--rel_emb", type=str, default="RelEmb.npy", help="filename to save relation embedding")
parser.add_argument("--batch_size", type=int, default=64, help="batch size used in inference")

def desc_dataset(path, dictionary):
    now_path=path
    dataset=data_utils.load_indexed_dataset(
        now_path,
        dictionary,
        None,
        combine=False,
    )
    dataset = PrependTokenDataset(dataset, 0)
    dataset = TruncateDataset(dataset, 512)
    dataset = RightPadDataset(dataset, pad_idx=1)
    return dataset

if __name__=='__main__':
    args = parser.parse_args()
    dictionary = Dictionary.load(args.dict)
    desc = desc_dataset(os.path.join(args.data, 'train'), dictionary)
    roberta = RobertaModel.from_pretrained(args.ckpt_dir, checkpoint_file = args.ckpt)
    roberta.eval()
    np.save(args.rel_emb, roberta.model.ke_heads['wikiData'].relation_emb.weight.cpu().detach().numpy())
    entity_embs=[]
    for i in range(0, len(desc)//args.batch_size+1):
        s = i*args.batch_size
        t = min((i+1)*args.batch_size, len(desc))
        if s >= len(desc):
            continue
        datas = [desc[x] for x in range(s,t)]
        embs = roberta.extract_features(desc.collater(datas))[:,0,:]
        entity_embs.append(embs.cpu().detach().numpy())
    a = np.concatenate(entity_embs, axis=0)
    np.save(args.ent_emb, a)