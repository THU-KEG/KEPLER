# coding=utf-8
"""do negative sampling and dump training data"""
import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime

parser=argparse.ArgumentParser()
parser.add_argument("--dumpPath", type=str, help="path to store output files, do NOT create it previously")
parser.add_argument("-ns", "--negative_sampling_size", type=int, default=1)
parser.add_argument("--train", type=str, help="file name of training triplets")
parser.add_argument("--valid", type=str, help="file name of validation triplets")
parser.add_argument("--ent_desc", type=str, help="path to the entity description file (after BPE encoding)")

def getTriples(path):
    res=[]
    with open(path, "r") as fin:
        lines=fin.readlines()
        for l in lines:
            tmp=[int(x) for x in l.split()]
            res.append((tmp[0],tmp[2],tmp[1]))
    return res

def count_frequency(triples, start=4):
    count = {}
    for head, relation, tail in triples:
        hr=",".join([str(head), str(relation)])
        tr=",".join([str(tail), str(-relation-1)])
        if hr not in count:
            count[hr] = start
        else:
            count[hr] += 1
        if tr not in count:
            count[tr] = start
        else:
            count[tr] += 1
    return count
    
def get_true_head_and_tail(triples):
    true_head = {}
    true_tail = {}

    for head, relation, tail in triples:
        if (head, relation) not in true_tail:
            true_tail[(head, relation)] = []
        true_tail[(head, relation)].append(tail)
        if (relation, tail) not in true_head:
            true_head[(relation, tail)] = []
        true_head[(relation, tail)].append(head)

    for relation, tail in true_head:
        true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
    for head, relation in true_tail:
        true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 
    return true_head, true_tail

def getTokens(s):
    return min(len(s.split()),512)

def genSample(triples, args, split, Qdesc, true_head, true_tail):
    fHead = open(os.path.join(args.dumpPath, "head", split)+".bpe", "w")
    fTail = open(os.path.join(args.dumpPath, "tail", split)+".bpe", "w")
    fnHead = open(os.path.join(args.dumpPath, "negHead", split)+".bpe", "w")
    fnTail = open(os.path.join(args.dumpPath, "negTail", split)+".bpe", "w")
    rel=[]
    sizes=[]
    nE=len(Qdesc)
    for h,r,t in triples:
        rel.append(r)
        fHead.write(Qdesc[h])
        fTail.write(Qdesc[t])
        size=getTokens(Qdesc[h])
        size+=getTokens(Qdesc[t])
        for mode in ["head-batch","tail-batch"]:
            negative_sample_list = []
            negative_sample_size = 0
            while negative_sample_size < args.negative_sampling_size:
                negative_sample = np.random.randint(nE, size=args.negative_sampling_size*2)
                if mode == 'head-batch':
                    mask = np.in1d(
                        negative_sample, 
                        true_head[(r, t)], 
                        assume_unique=True, 
                        invert=True
                    )
                elif mode == 'tail-batch':
                    mask = np.in1d(
                        negative_sample, 
                        true_tail[(h, r)], 
                        assume_unique=True, 
                        invert=True
                    )
                negative_sample = negative_sample[mask]
                negative_sample_list.append(negative_sample)
                negative_sample_size += negative_sample.size
            negative_sample = np.concatenate(negative_sample_list)[:args.negative_sampling_size]
            for t in negative_sample:
                x=int(t)
                if mode == 'head-batch':
                    fnHead.write(Qdesc[x])
                else:
                    fnTail.write(Qdesc[x])
                size += getTokens(Qdesc[x])
        sizes.append(size)
    fHead.close()
    fTail.close()
    fnHead.close()
    fnTail.close()
    np.save(os.path.join(args.dumpPath, "relation", split)+".npy", np.array(rel))
    np.save(os.path.join(args.dumpPath, "sizes", split)+".npy", np.array(sizes))

if __name__=='__main__':
    args=parser.parse_args()
    TrainTriples = getTriples(args.train)
    ValidTriples = getTriples(args.valid)
    AllTriples = TrainTriples + ValidTriples
    Qdesc=[]
    with open(args.ent_desc, "r") as fin:
        Qdesc=fin.readlines()
    print(str(datetime.now())+" load finish")
    count = count_frequency(AllTriples)
    true_head, true_tail = get_true_head_and_tail(AllTriples)
    os.mkdir(args.dumpPath)
    json.dump(count, open(os.path.join(args.dumpPath, "count.json"), "w"))
    for nm in ["head","tail","negHead","negTail","relation","sizes"]:
        os.mkdir(os.path.join(args.dumpPath, nm))
    print(str(datetime.now()) + " preparation finished")
    genSample(TrainTriples, args, "train", Qdesc, true_head, true_tail)
    print(str(datetime.now())+" training set finished")
    genSample(ValidTriples, args, "valid", Qdesc, true_head, true_tail)
    print(str(datetime.now())+" all finished")