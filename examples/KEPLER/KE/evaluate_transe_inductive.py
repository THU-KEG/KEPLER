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

    parser.add_argument('--dataset', help="test dataset")
    args = parser.parse_args()
    
    # Building the graph 
    app = gap.KnowledgeGraphApplication(dim=args.dim)
    app.load(file_name=args.dataset)
    app.build()
    app.train(model='TransE', num_epoch=0)

    gv_entity2id = app.graph.entity2id
    gv_relation2id = app.graph.relation2id

    # Load embeddings (Only load the embeddings that appear in the entity2id file)
    entity_embeddings_full = np.load(args.entity_embeddings)
    relation_embeddings_full = np.load(args.relation_embeddings)
    entity2id_ori = json.load(open(args.entity2id))
    relation2id_ori = json.load(open(args.relation2id))

    entity_embeddings = np.zeros((len(gv_entity2id), args.dim), dtype=np.float32) 
    entity2id = {}
    i = 0
    for key in tqdm(gv_entity2id):
        entity2id[key] = i
        entity_embeddings[i] = entity_embeddings_full[entity2id_ori[key]]
        i += 1

    relation_embeddings = np.zeros((len(gv_relation2id), args.dim), dtype=np.float32) 
    relation2id = {}
    i = 0
    for key in tqdm(gv_relation2id):
        relation2id[key] = i
        relation_embeddings[i] = relation_embeddings_full[relation2id_ori[key]]
        i += 1
    
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
    app.evaluate('link_prediction', file_name=args.dataset, filter_files=[args.dataset])

if __name__ == '__main__':
    main()
