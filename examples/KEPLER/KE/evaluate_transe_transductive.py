import argparse
import graphvite as gv
import graphvite.application as gap
import numpy as np
import json
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity_embeddings', help='numpy of entity embeddings')
    parser.add_argument('--relation_embeddings', help='numpy of relation embeddings')
    parser.add_argument('--entity2id', help='entity name to numpy id json')
    parser.add_argument('--relation2id', help='entity name to numpy id json')
    parser.add_argument('--dim', type=int, help='size of embedding')

    parser.add_argument('--train_dataset', help="train dataset")
    parser.add_argument('--val_dataset', help="val dataset")
    parser.add_argument('--dataset', help="test dataset")
    args = parser.parse_args()
    
    # Load embeddings (Only load the embeddings that appear in the entity2id file)
    entity_embeddings_full = np.load(args.entity_embeddings)
    relation_embeddings = np.load(args.relation_embeddings)
    entity2id_ori = json.load(open(args.entity2id))
    relation2id = json.load(open(args.relation2id))

    entity_embeddings = np.zeros((len(entity2id_ori), args.dim), dtype=np.float32) 
    entity2id = {}
    i = 0
    for key in tqdm(entity2id_ori):
        entity2id[key] = i
        entity_embeddings[i] = entity_embeddings_full[entity2id_ori[key]]
        i += 1
    
    # Building the graph 
    app = gap.KnowledgeGraphApplication(dim=args.dim)
    app.load(file_name=args.train_dataset)
    app.build()
    app.train(model='TransE', num_epoch=0)

    # Load embeddings to graphvite
    print('load data ......')
    assert(len(relation_embeddings) == len(app.solver.relation_embeddings))
    assert(len(entity_embeddings) == len(app.solver.entity_embeddings))
    app.solver.relation_embeddings[:] = relation_embeddings
    print('loaded relation embeddings')
    app.solver.entity_embeddings[:] = entity_embeddings
    print('loaded entity embeddings')
    
    # (Modified gv) Replace mapping with our own
    app.entity2id = entity2id
    app.relation2id = relation2id
    
    print('start evaluation ......')
    app.evaluate('link_prediction', file_name=args.dataset, filter_files=[args.train_dataset, args.val_dataset, args.dataset])

if __name__ == '__main__':
    main()
