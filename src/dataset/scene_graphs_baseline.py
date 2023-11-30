import pandas as pd
import torch
from torch.utils.data import Dataset


model_name = 'sbert'
path = 'dataset/scene_graphs'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'


class SceneGraphsBaselineDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.prompt = 'Please answer the given question.'
        self.graph = None
        self.graph_type = 'Scene Graph'
        self.questions = pd.read_csv(f'{path}/questions.csv')

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.questions)

    def __getitem__(self, index):
        data = self.questions.iloc[index]
        image_id = data['image_id']
        question = f'Question: {data["question"]}\n\nAnswer:'
        nodes = pd.read_csv(f'{path_nodes}/{image_id}.csv')
        edges = pd.read_csv(f'{path_edges}/{image_id}.csv')
        graph = torch.load(f'{path_graphs}/{image_id}.pt')
        desc = nodes.to_csv(index=False)+'\n'+edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])

        return {
            'id': index,
            'question': question,
            'label': data['answer'],
            'desc': desc,
            'graph': graph,
        }

    def get_idx_split(self):

        # Load the saved indices
        with open(f'{path}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}


if __name__ == '__main__':
    dataset = SceneGraphsBaselineDataset()

    data = dataset[0]
    for k, v in data.items():
        print(f'{k}: {v}')

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
