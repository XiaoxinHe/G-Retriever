import torch
import pandas as pd
import os
import networkx as nx
from tqdm import tqdm

from torch_geometric.utils.subgraph import subgraph
from torch_geometric.data import Data
import torch
import pandas as pd


def load_umls_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf8") as fin:
        vocab = [l.strip() for l in fin]
    df_vocab = pd.DataFrame([v.split('\t')for v in vocab], columns=['concept_id', 'concept'])
    return df_vocab


def load_gounded(path='dataset/tape_medqa/all.grounded.jsonl'):
    print('Loading grounded data from ', path)
    df = pd.read_json(path_or_buf=path, lines=True)
    return df


def load_adj(path='dataset/tape_medqa/all.graph.adj.pk'):
    # load networkx data
    print('Loading networkx data from ', path)
    nx_data = nx.read_gpickle(path)
    print('# of samples: ', len(nx_data))
    return nx_data


def load_umls(path='dataset/umls/pyg_umls_graph.pt'):
    print('Loading UMLS graph...')
    umls_graph = torch.load(path)
    print(umls_graph)
    return umls_graph


def merge_adj():
    print("Merging adj...")
    data = []
    for split in ['train', 'dev', 'test']:
        temp = load_adj(f'dataset/medqa/graph/{split}.graph.adj.pk')
        data.extend(temp)

    print("Writing adj to dataset/tape_medqa/all.graph.adj.pk")
    nx.write_gpickle(data, f'dataset/tape_medqa/all.graph.adj.pk')
    return data


def merge_grounded():
    print("Merging grounded...")
    df = []
    for split in ['train', 'dev', 'test']:
        file_path = f'dataset/medqa/grounded/{split}.grounded.jsonl'
        temp = load_gounded(path=file_path)
        df.append(temp)

    print("Writing grounded to dataset/tape_medqa/all.grounded.jsonl")
    df = pd.concat(df)
    df.to_json('dataset/tape_medqa/all.grounded.jsonl', orient='records', lines=True)
    return df


def load_statement(path='dataset/tape_medqa/all.statement.jsonl'):
    print('Loading statement from ', path)
    df = pd.read_json(path_or_buf=path, lines=True)
    return df


def merge_statement():
    print('Merging statement...')
    train = load_statement('dataset/medqa/statement/train.statement.jsonl')
    dev = load_statement('dataset/medqa/statement/dev.statement.jsonl')
    test = load_statement('dataset/medqa/statement/test.statement.jsonl')

    # generaye split indices
    num_samples = [len(train), len(dev), len(test)]

    train_indices = list(range(num_samples[0]))
    val_indices = list(range(num_samples[0], num_samples[0] + num_samples[1]))
    test_indices = list(range(num_samples[0] + num_samples[1], num_samples[0] + num_samples[1] + num_samples[2]))

    path = 'dataset/tape_medqa/split'
    os.makedirs(path, exist_ok=True)

    # Save the indices to separate files
    with open(f'{path}/train_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, train_indices)))

    with open(f'{path}/val_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, val_indices)))

    with open(f'{path}/test_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, test_indices)))

    print("# train: {}, val: {}, test: {}".format(len(train_indices), len(val_indices), len(test_indices)))

    print('Writing statement to dataset/tape_medqa/all.statement.jsonl')
    df = pd.concat([train, dev, test])
    df.to_json('dataset/tape_medqa/all.statement.jsonl', orient='records', lines=True)
    return df


def generate_pyg_data(data):
    q_concepts = []
    a_concepts = []
    subset = []
    for d in data:
        adj, concepts, qmask, amask = d
        qc = concepts[qmask]
        ac = concepts[amask]
        q_concepts.extend(qc.tolist())
        a_concepts.extend(ac.tolist())
        for c in concepts.tolist():
            if c not in subset:
                subset.append(c)

    q_concepts = list(set(q_concepts))
    a_concepts = list(set(a_concepts))
    q_mask = [i in q_concepts for i in subset]
    a_mask = [i in a_concepts for i in subset]

    subset = torch.tensor(subset).long()
    edge_index, edge_attr = subgraph(subset=subset, edge_index=umls_graph.edge_index, relabel_nodes=False,
                                     num_nodes=umls_graph.num_nodes, edge_attr=umls_graph.edge_attr)

    # relabel nodes
    mapping = dict(zip(subset.tolist(), range(len(subset))))
    edge_index = torch.tensor([[mapping[edge_index[0, i].item()], mapping[edge_index[1, i].item()]] for i in range(edge_index.shape[1])]).T

    x = torch.Tensor(umls_graph.x[subset])
    qmask = torch.tensor(qmask).bool()
    amask = torch.tensor(amask).bool()
    concepts = torch.tensor(concepts).long()
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=subset.size(0), q_mask=q_mask, a_mask=a_mask, concepts=subset)
    return pyg_data


def generate_questions():
    print('Generating questions...')
    df_stat = pd.read_json('dataset/tape_medqa/all.statement.jsonl', lines=True)

    data = []
    for index, row in tqdm(df_stat.iterrows(), total=df_stat.shape[0]):
        res = {}
        question = row['question']['stem']
        choices = [c['label']+'. '+c['text'] for c in row['question']['choices']]
        text_choices = ('\n').join(choices)
        question = f'Question: {question}\nChoices:\n{text_choices}'

        res['question'] = question
        res['label'] = row['answerKey']
        data.append(res)

    print("Saving questions to dataset/tape_medqa/medqa.csv")
    pd.DataFrame(data).to_csv('dataset/tape_medqa/medqa.csv', index=False)


def generate_desc():
    print("Generating description...")

    os.makedirs('dataset/tape_medqa/text', exist_ok=True)

    vocab = load_umls_vocab('dataset/umls/concept_names.txt')
    question_label = pd.read_csv('dataset/tape_medqa/medqa.csv')

    for index, row in tqdm(question_label.iterrows(), total=question_label.shape[0]):

        pyg_graph = torch.load(f'dataset/tape_medqa/graph/{index}.pt')
        desc = ""
        for c in vocab.iloc[pyg_graph.concepts[pyg_graph.a_mask]]['concept']:
            desc += f'{c}: <NodeEmb>\n'
        with open(f'dataset/tape_medqa/text/{index}.txt', 'w') as f:
            f.write(desc)

    print("Done!")


def preprocess():
    global umls_graph
    umls_graph = load_umls()

    df_stat = merge_statement()
    nx_data = merge_adj()

    os.makedirs(f'dataset/tape_medqa/graph/', exist_ok=True)

    options = 4
    assert len(nx_data) == len(df_stat) * options
    for index, row in tqdm(df_stat.iterrows(), total=len(df_stat)):
        data = nx_data[index*options:(index+1)*options]
        pyg_data = generate_pyg_data(data)
        torch.save(pyg_data, f'dataset/tape_medqa/graph/{index}.pt')


if __name__ == '__main__':

    # preprocess()

    # generate_questions()

    generate_desc()
