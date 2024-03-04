import os
import json
import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric.data.data import Data
from src.utils.lm_modeling import load_model, load_text2embedding
from src.dataset.preprocess.generate_split import generate_split


model_name = 'sbert'
path = 'dataset/scene_graphs'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'


def textualize_graph(data):
    # mapping from object id to index
    objectid2nodeid = {object_id: idx for idx, object_id in enumerate(data['objects'].keys())}
    nodes = []
    edges = []
    for objectid, object in data['objects'].items():
        # nodes
        node_attr = f'name: {object["name"]}'
        x, y, w, h = object['x'], object['y'], object['w'], object['h']
        if len(object['attributes']) > 0:
            node_attr = node_attr + '; attribute: ' + (', ').join(object["attributes"])
        node_attr += '; (x,y,w,h): ' + str((x, y, w, h))
        nodes.append({'node_id': objectid2nodeid[objectid], 'node_attr': node_attr})

        # edges
        for rel in object['relations']:
            src = objectid2nodeid[objectid]
            dst = objectid2nodeid[rel['object']]
            edge_attr = rel['name']
            edges.append({'src': src, 'edge_attr': edge_attr, 'dst': dst})

    return nodes, edges


def step_one():
    dataset = json.load(open('dataset/gqa/sceneGraphs/train_sceneGraphs.json'))

    os.makedirs(path_nodes, exist_ok=True)
    os.makedirs(path_edges, exist_ok=True)

    for imageid, object in tqdm(dataset.items(), total=len(dataset)):
        node_attr, edge_attr = textualize_graph(object)
        pd.DataFrame(node_attr, columns=['node_id', 'node_attr']).to_csv(f'{path_nodes}/{imageid}.csv', index=False)
        pd.DataFrame(edge_attr, columns=['src', 'edge_attr', 'dst']).to_csv(f'{path_edges}/{imageid}.csv', index=False)


def step_two():
    def _encode_questions():
        q_embs = text2embedding(model, tokenizer, device, df.question.tolist())
        torch.save(q_embs, f'{path}/q_embs.pt')

    def _encode_graphs():
        image_ids = df.image_id.unique()
        for i in tqdm(image_ids):
            nodes = pd.read_csv(f'{path_nodes}/{i}.csv')
            edges = pd.read_csv(f'{path_edges}/{i}.csv')
            node_attr = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
            edge_attr = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
            edge_index = torch.tensor([edges.src, edges.dst]).long()
            pyg_graph = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))
            torch.save(pyg_graph, f'{path_graphs}/{i}.pt')

    df = pd.read_csv(f'{path}/questions.csv')
    os.makedirs(path_graphs, exist_ok=True)
    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    _encode_questions()
    _encode_graphs()


if __name__ == '__main__':
    step_one()
    step_two()
    generate_split(100000, f'{path}/split')
