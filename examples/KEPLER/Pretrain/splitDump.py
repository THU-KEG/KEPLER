import os
import sys
import json
import argparse
import numpy as np
import random
from shutil import copyfile
from datetime import datetime

parser=argparse.ArgumentParser()
parser.add_argument("--Path", type=str, help="path to preprocessed KG data")
parser.add_argument("--split_size", type=int, help="max number of instance in each split")
parser.add_argument("--negative_sampling_size", type=int, help="negative sample size")

def nsIdx(args,idx):
    res=[]
    for x in idx:
        for i in range(0, args.negative_sampling_size):
            res.append(x*args.negative_sampling_size+i)
    return res

def spBpe(lines,idx,fout):
    for i in idx:
        fout.write(lines[i])
    fout.close()

if __name__ == '__main__':
    args = parser.parse_args()
    head = open(os.path.join(args.Path, "head", "train.bpe"),"r")
    lines = head.readlines()
    Idx = np.random.permutation(len(lines)) #random shuffle the data among splits
    SNUM = len(lines)//args.split_size + (1 if len(lines) % args.split_size>0 else 0)
    print("The data will be splited into %d splits"%(SNUM))
    for i in range(0,SNUM):
        nPath=args.Path+"_%d"%(i)
        os.mkdir(nPath)
        for nm in ["head", "tail", "negHead", "negTail", "relation", "sizes"]:
            os.mkdir(os.path.join(nPath,nm))
            if nm in ["relation", "sizes"]:
                copyfile(os.path.join(args.Path, nm, "valid.npy"), os.path.join(nPath, nm, "valid.npy"))
            else:
                copyfile(os.path.join(args.Path, nm, "valid.bpe"), os.path.join(nPath, nm, "valid.bpe"))
        copyfile(os.path.join(args.Path, "count.json"), os.path.join(nPath, "count.json"))
    for nm in ["head","tail"]:
        lines = open(os.path.join(args.Path, nm, "train.bpe"), "r").readlines()
        for i in range(0,SNUM):
            nPath = os.path.join(args.Path+"_%d"%(i), nm, "train.bpe")
            sIdx = Idx[i*args.split_size:min((i+1)*args.split_size, len(Idx))]
            spBpe(lines, sIdx, open(nPath,"w"))
    for nm in ["negHead","negTail"]:
        lines = open(os.path.join(args.Path, nm, "train.bpe"),"r").readlines()
        for i in range(0, SNUM):
            nPath = os.path.join(args.Path+"_%d"%(i), nm, "train.bpe")
            sIdx = nsIdx(args, Idx[i*args.split_size:min((i+1)*args.split_size, len(Idx))])
            spBpe(lines, sIdx, open(nPath, "w"))
    for nm in ["relation","sizes"]:
        lines = np.load(os.path.join(args.Path,nm,"train.npy"))
        for i in range(0,SNUM):
            nPath = os.path.join(args.Path+"_%d"%(i), nm, "train.npy")
            sIdx = Idx[i*args.split_size:min((i+1)*args.split_size, len(Idx))]
            np.save(nPath, lines[sIdx])