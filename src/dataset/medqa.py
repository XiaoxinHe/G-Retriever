import torch
import pandas as pd
from torch.utils.data import Dataset


class MedQADataset(Dataset):
    def __init__(self):
        super().__init__()

        self.data = pd.read_csv('dataset/tape_medqa/medqa.csv')

        self.prompt = "\nThe correct answer is "
        self.graph = None
        self.graph_type = 'Medical Knowledge Graph'

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        data = self.data.iloc[index]

        # load the scene graph on the fly
        pyg_graph = torch.load('dataset/tape_medqa/graph/{}.pt'.format(index))
        with open('dataset/tape_medqa/text/{}.txt'.format(index)) as f:
            desc = f.readlines()

        return {
            'id': index,
            'question': data['question'],
            'label': data['label'],
            'graph': pyg_graph,
            'desc': desc,
        }

    def get_idx_split(self):

        # Load the saved indices
        with open('dataset/tape_medqa/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open('dataset/tape_medqa/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open('dataset/tape_medqa/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}


if __name__ == '__main__':
    dataset = MedQADataset()

    data = dataset[0]
    for k, v in data.items():
        print(f'{k}: {v}')

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
