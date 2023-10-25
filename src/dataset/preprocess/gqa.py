import os
import torch
import json
import gensim
import pandas as pd
import numpy as np

from tqdm import tqdm
from torch_geometric.data.data import Data
from src.dataset.preprocess.generate_split import generate_split


hidden_dim = 300
NUM_SAMPLES = 100*1000


def encode_text_with_word2vec(model, text):
    words = text.split()  # Tokenize the text into words
    word_vectors = []

    for word in words:
        try:
            vector = model[word]  # Get the Word2Vec vector for the word
            word_vectors.append(vector)
        except KeyError:
            # Handle the case where the word is not in the vocabulary
            pass

    if word_vectors:
        # Calculate the mean of word vectors to represent the text
        text_vector = sum(word_vectors) / len(word_vectors)
    else:
        # Handle the case where no word vectors were found
        text_vector = np.zeros(hidden_dim)

    return text_vector


def encode_attributes(model, text):
    emb = [encode_text_with_word2vec(model, t) for t in text]
    return torch.Tensor(emb)


def construct_graph(scene_graph, model):
    scene_graph, mapping = simplify_scene_graph(scene_graph)

    node_attr = []
    edge_attr = []
    edges = []
    desc = []

    for idx, object in scene_graph['objects'].items():
        name = object['name']
        _node_attr = name

        attr = ""
        for a in object['attributes']:
            attr = attr+f"{a}, "
            _node_attr += f" {a}"

        rel = ""
        for r in object['relations']:
            rel = rel+f"({r['name']}, {int(r['object'])}), "
            edges.append((int(idx), int(r['object'])))
            edge_attr.append(r['name'])

        d = f"{idx}: info: <NodeEmb>; name: {name}; "
        if attr:
            d = d+f"attributes: [{attr[:-2]}]; "
        if rel:
            d = d+f"relations: [{rel[:-2]}]"
        desc.append(d.strip())
        node_attr.append(_node_attr.strip())

    x = encode_attributes(model, node_attr)
    e = encode_attributes(model, edge_attr)
    data = Data(x=x,
                edge_index=torch.tensor(edges).T,
                edge_attr=e,
                num_nodes=len(desc))
    return data, desc


def simplify_scene_graph(scene_graph):
    objects = scene_graph['objects']
    mapping = {object_id: str(idx)
               for idx, object_id in enumerate(objects.keys())}

    str_objects = json.dumps(objects)

    for object_id, idx in mapping.items():
        str_objects = str_objects.replace(object_id, idx)
    object = json.loads(str_objects)
    object = {int(k): v for k, v in object.items()}
    scene_graph['objects'] = object
    return scene_graph, mapping


def write2file(fn, node_desc):
    with open(fn, 'w') as f:
        for line in node_desc:
            f.writelines(line)
            f.writelines('\n')


def process_questions():
    print('Start processing questions...')
    with open('dataset/gqa/questions/train_balanced_questions.json') as f:
        questions = json.load(f)
    data = []
    for qid, question in questions.items():
        data.append({'q_id': qid,
                    'image_id': question['imageId'],
                     'question': question['question'],
                     'answer': question['answer'],
                     'full_answer': question['fullAnswer']})
    df = pd.DataFrame(data)

    np.random.seed(42)  # Numpy module.
    q_ids = np.arange(len(questions))
    np.random.shuffle(q_ids)
    df = df.iloc[q_ids[:NUM_SAMPLES]]
    os.makedirs('dataset/tape_gqa', exist_ok=True)
    df.to_csv('dataset/tape_gqa/questions.csv', index=False)

    print('Finish processing questions!')


def process_scene_graphs():
    # Load Google's pre-trained Word2Vec model.
    print('Loading Google\'s pre-trained Word2Vec model...')
    path = '/home/xiaoxin/word2vec/GoogleNews-vectors-negative300.bin.gz'
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    print('Finish loading word2vec model!')

    # process scene graphs
    g_filename = 'dataset/gqa/sceneGraphs/train_sceneGraphs.json'
    out_dir_text = 'dataset/tape_gqa/scene_graphs_text'
    out_dir_graph = 'dataset/tape_gqa/scene_graphs_pyg'

    os.makedirs(out_dir_text, exist_ok=True)
    os.makedirs(out_dir_graph, exist_ok=True)

    print('Start processing scene graphs...')
    data = json.load(open(g_filename))
    for graph_id, graph in tqdm(data.items()):
        graph, node_desc = construct_graph(graph, model)
        write2file(f'{out_dir_text}/{graph_id}.txt', node_desc)
        torch.save(graph, f"{out_dir_graph}/{graph_id}.pt")
    print('Finish processing scene graphs!')


def main():
    process_scene_graphs()

    process_questions()

    # process split
    generate_split(NUM_SAMPLES, 'dataset/tape_gqa/split')


if __name__ == '__main__':
    main()
