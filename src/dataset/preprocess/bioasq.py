from torch_geometric.utils import subgraph
import re
import numpy as np

import os
import json
from tqdm import tqdm
import pandas as pd

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph

from collections import OrderedDict
from src.dataset.preprocess.generate_split import generate_split

token_pattern = re.compile(r"(?u)\b\w+\b")

concept2id = None
concept2name = None
id2concept = None
relation2id = None
id2relation = None


def extract_yesno(fn='dataset/bioasq_all/raw/training12b_new.json'):

    data = json.load(open(fn))
    df = pd.DataFrame(data['questions'])
    print("# samples: ", len(df))
    df = df[df.type == 'yesno']
    df['ideal_answer'] = df['ideal_answer'].apply(lambda x: x[0])
    yesno = pd.DataFrame(zip(df.body, df.exact_answer, df.ideal_answer,), columns=['question', 'exact_answer', 'ideal_answer'])
    print("# yesno samples: ", len(yesno))

    yesno.to_csv('dataset/bioasq_all/yesno.csv', index=False)
    cnt = yesno.exact_answer.value_counts()
    print(f"Yes: {cnt['yes']} ({(cnt['yes']/(cnt['yes']+cnt['no'])*100).round(2)}%)")
    print(f"No: {cnt['no']} ({(cnt['no']/(cnt['yes']+cnt['no'])*100).round(2)}%)")
    return yesno


def load_umls_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf8") as fin:
        vocab = [l.strip() for l in fin]
    return vocab


def load_resources():
    global concept2id, id2concept, relation2id, id2relation, concept2name
    id2concept = [w.strip() for w in open('dataset/umls/concepts.txt')]
    concept2id = {w: i for i, w in enumerate(id2concept)}
    concept2name = {}
    for line in open('dataset/umls/concept_names.txt'):
        c, n = line.strip().split('\t')
        concept2name[c] = n
    id2relation = [r.strip() for r in open('dataset/umls/relations.txt')]
    relation2id = {r: i for i, r in enumerate(id2relation)}


def sent2glove(sent):
    words = token_pattern.findall(sent.lower())
    vec = np.sum([glove_w2v.get(w, np.zeros((50,), dtype=float)) for w in words], axis=0)
    if not isinstance(vec, np.ndarray):
        vec = np.zeros((50,), dtype=float)
    l2norm = np.sqrt((vec ** 2).sum())
    vec = vec / (l2norm + 1e-8)
    return vec


def load_glove():
    global glove_w2v, id2glove

    print('Loading glove...')
    glove_w2v = {}
    for line in tqdm(open('dataset/glove/glove.6B/glove.6B.50d.txt')):
        elms = line.split()
        glove_w2v[elms[0]] = np.array(elms[1:], dtype=float)
    print('Loaded glove.')

    print('Mapping concepts to glove vecs...')
    global concept2id, id2concept, relation2id, id2relation, concept2name
    if concept2id is None:
        load_resources()
    id2glove = []
    for id, concept in enumerate(tqdm(id2concept)):
        name = concept2name[concept]
        # name = name.replace('-', ' ') #no need as token_pattern.findall() will handle
        # name = name.replace(',', ' ')
        name = name.replace('_', ' ')
        id2glove.append(sent2glove(name))
    print('Mapped concepts to glove vecs.')


def get_glove_score(cids, question):
    if len(cids) == 0:
        return {}
    sent_vec = sent2glove(question)  # [dim,]
    concept_vecs = np.stack([id2glove[cid] for cid in cids])  # [nodes, dim]
    scores = list(concept_vecs.dot(sent_vec))  # [nodes,]
    assert len(scores) == len(cids)
    # score: from high to low
    cid2score = OrderedDict(
        sorted(list(zip(cids, scores)), key=lambda x: -x[1]))
    return cid2score


def generate_graph(data, max_num_nodes):
    cids = [concept2id[c] for c in data.qc]
    subset, edge_index, inv, edge_mask = k_hop_subgraph(
        node_idx=cids, num_hops=2, edge_index=umls_graph.edge_index, relabel_nodes=True, num_nodes=umls_graph.num_nodes)
    extra_nodes = list(set(subset.tolist())-set(subset[inv].tolist()))

    # prune extra nodes
    cid2score = get_glove_score(extra_nodes, data.sent)
    selected_extra_nodes = list(cid2score.keys())[:max_num_nodes]
    selected_subset = subset[inv].tolist() + selected_extra_nodes

    num_nodes = len(selected_subset)
    edge_index, edge_attr = subgraph(subset=selected_subset,
                                     edge_index=umls_graph.edge_index,
                                     relabel_nodes=True,
                                     num_nodes=umls_graph.num_nodes,
                                     edge_attr=umls_graph.edge_attr)

    # convert to pyg data
    label = {'yes': 1, 'no': 0}
    x = torch.Tensor(umls_graph.x[selected_subset])
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=label[data.exact_answer])

    desc = []
    for i in range(num_nodes):
        name = concept2name[id2concept[selected_subset[i]]]
        temp = f'{i}: info: <NodeEmb>; name: {name}'
        row, col = pyg_data.edge_index
        edge_mask = (row == i)
        relations = []
        for dst, e in zip(col[edge_mask].tolist(), pyg_data.edge_attr[edge_mask].tolist()):
            if e >= len(id2relation):
                continue
            relations.append(f'({id2relation[e]}, {dst})')
        if len(relations) > 0:
            relations = (', ').join(relations)
            temp += f'; relations: [{relations}]'
        temp += '\n'
        desc.append(temp)

    return pyg_data, desc


def generate_graph_and_text(df_bioasq, max_num_nodes):
    print('Generating graph and text...')

    os.makedirs('dataset/tape_bioasq/graph', exist_ok=True)
    os.makedirs('dataset/tape_bioasq/text', exist_ok=True)

    for index, row in tqdm(df_bioasq.iterrows(), total=len(df_bioasq)):
        pyg_data, desc = generate_graph(row, max_num_nodes=max_num_nodes)
        torch.save(pyg_data, f'dataset/tape_bioasq/graph/{index}.pt')
        with open(f'dataset/tape_bioasq/text/{index}.txt', 'w') as f:
            for line in desc:
                f.write(line)
    print('Generated graph and text.')


def main():

    df_bioasq = pd.read_json('dataset/tape_bioasq/yesno.json', orient='records', lines=True)

    global umls_graph
    umls_graph = torch.load('dataset/umls/pyg_umls_graph.pt')

    load_glove()

    generate_graph_and_text(df_bioasq, max_num_nodes=300)

    generate_split(len(df_bioasq), 'dataset/tape_bioasq/split')


if __name__ == '__main__':
    main()
