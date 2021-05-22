import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str, help="path to original text file")
parser.add_argument("--train", type=str, help="path to original training data file")
parser.add_argument("--valid", type=str, help="path to original validation data file")
parser.add_argument("--converted_text", type=str, default="Qdesc.txt", help="path to converted text file")
parser.add_argument("--converted_train", type=str, default="train.txt", help="path to converted training file")
parser.add_argument("--converted_valid", type=str, default="valid.txt", help="path to converted validation file")

if __name__=='__main__':
    args = parser.parse_args()
    Qid={}  #Entity to id (line number in the description file)
    Pid={}  #Relation to id
    def getNum(s):
        return int(s[1:])
    with open(args.text, "r") as fin:
        with open(args.converted_text, "w") as fout:
            lines = fin.readlines()
            Cnt=0
            for idx, line in enumerate(lines):
                data = line.split('\t')
                assert len(data) >= 2
                assert data[0].startswith('Q')
                desc = '\t'.join(data[1:]).strip()
                if getNum(data[0])>1000:
                    continue
                fout.write(desc+"\n")
                Qid[data[0]] = Cnt#idx
                Cnt+=1
    def convert_triples(inFile, outFile):
        with open(inFile, "r") as fin:
            with open(outFile, "w") as fout:
                lines = fin.readlines()
                for line in lines:
                    data = line.strip().split('\t')
                    assert len(data) == 3
                    if getNum(data[0])>1000 or getNum(data[2]) > 1000:
                        continue
                    if data[1] not in Pid:
                        Pid[data[1]] = len(Pid)
                    fout.write("%d %d %d\n"%(Qid[data[0]], Pid[data[1]], Qid[data[2]]))
    convert_triples(args.train, args.converted_train)
    convert_triples(args.valid, args.converted_valid)
