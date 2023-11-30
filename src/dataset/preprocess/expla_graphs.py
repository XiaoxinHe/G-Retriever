import re
import os
import torch
import pandas as pd

from tqdm import tqdm
from torch_geometric.data.data import Data

from src.dataset.preprocess.generate_split import generate_split
from src.utils.lm_modeling import load_model, load_text2embedding


model_name = 'sbert'
path = 'dataset/expla_graphs'
dataset = pd.read_csv(f'{path}/train_dev.tsv', sep='\t')


def textualize_graph(graph):
    triplets = re.findall(r'\((.*?)\)', graph)
    nodes = {}
    edges = []
    for tri in triplets:
        src, edeg_attr, dst = tri.split(';')
        src = src.lower().strip()
        dst = dst.lower().strip()
        if src not in nodes:
            nodes[src] = len(nodes)
        if dst not in nodes:
            nodes[dst] = len(nodes)
        edges.append({'src': nodes[src], 'edge_attr': edeg_attr.lower().strip(), 'dst': nodes[dst], })

    nodes = pd.DataFrame(nodes.items(), columns=['node_attr', 'node_id'])
    edges = pd.DataFrame(edges)
    return nodes, edges


def step_one():
    # generate textual graphs
    os.makedirs(f'{path}/nodes', exist_ok=True)
    os.makedirs(f'{path}/edges', exist_ok=True)

    for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
        nodes, edges = textualize_graph(row['graph'])
        nodes.to_csv(f'{path}/nodes/{i}.csv', index=False, columns=['node_id', 'node_attr'])
        edges.to_csv(f'{path}/edges/{i}.csv', index=False, columns=['src', 'edge_attr', 'dst'])


def step_two():

    def _encode_graph():
        print('Encoding graphs...')
        os.makedirs(f'{path}/graphs', exist_ok=True)
        for i in tqdm(range(len(dataset))):
            nodes = pd.read_csv(f'{path}/nodes/{i}.csv')
            edges = pd.read_csv(f'{path}/edges/{i}.csv')
            x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
            e = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
            edge_index = torch.LongTensor([edges.src, edges.dst])
            data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(nodes))
            torch.save(data, f'{path}/graphs/{i}.pt')

    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    _encode_graph()


if __name__ == '__main__':
    step_one()
    step_two()
    generate_split(len(dataset), f'{path}/split')
