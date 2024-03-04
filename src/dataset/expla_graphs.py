import json
import pandas as pd
import torch
from torch.utils.data import Dataset


PATH = 'dataset/expla_graphs'


class ExplaGraphsDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.text = pd.read_csv(f'{PATH}/train_dev.tsv', sep='\t')
        self.prompt = 'Question: Do argument 1 and argument 2 support or counter each other? Answer in one word in the form of \'support\' or \'counter\'.\n\nAnswer:'
        self.graph = None
        self.graph_type = 'Explanation Graph'

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):

        text = self.text.iloc[index]
        graph = torch.load(f'{PATH}/graphs/{index}.pt')
        question = f'Argument 1: {text.arg1}\nArgument 2: {text.arg2}\n{self.prompt}'
        nodes = pd.read_csv(f'{PATH}/nodes/{index}.csv')
        edges = pd.read_csv(f'{PATH}/edges/{index}.csv')
        desc = nodes.to_csv(index=False)+'\n'+edges.to_csv(index=False)

        return {
            'id': index,
            'label': text['label'],
            'desc': desc,
            'graph': graph,
            'question': question,
        }

    def get_idx_split(self):

        # Load the saved indices
        with open(f'{PATH}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]

        with open(f'{PATH}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]

        with open(f'{PATH}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}


if __name__ == '__main__':
    dataset = ExplaGraphsDataset()

    print(dataset.prompt)

    data = dataset[0]
    for k, v in data.items():
        print(f'{k}: {v}')

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
