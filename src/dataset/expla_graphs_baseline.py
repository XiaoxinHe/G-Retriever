import json
import pandas as pd
import torch
from torch.utils.data import Dataset


TAPE_PATH = 'dataset/tape_expla_graphs_baseline'


class ExplaGraphsBaselineDataset(Dataset):
    def __init__(self,):
        super().__init__()

        self.text = pd.read_csv(f'{TAPE_PATH}/train_dev.tsv', sep='\t')
        self.prompt = 'Given the explanation graph, do argument 1 and argument 2 support or counter each other? Answer in one word in the form of \'support\' or \'counter\'.\n\nAnswer:'
        self.graph = None
        self.graph_type = 'Explanation Graph'

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):

        text = self.text.iloc[index]
        question = f'Arguement 1: {text.arg1}\nArguement 2: {text.arg2}\n{self.prompt}'
        pyg_data = torch.load(f'{TAPE_PATH}/graph/{index}.pt')

        with open(f'{TAPE_PATH}/text/{index}.txt', 'r') as f:
            desc = f.readlines()
            desc = ''.join(desc)
            desc = f'{self.graph_type}: \n{desc}\n'

        return {
            'id': index,
            'label': text['label'],
            'desc': desc,
            'question': question,
            'graph': pyg_data
        }

    def get_idx_split(self):

        # Load the saved indices
        with open(f'{TAPE_PATH}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]

        with open(f'{TAPE_PATH}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]

        with open(f'{TAPE_PATH}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]
        return {'train': train_indices, 'val': val_indices, 'test': test_indices}


if __name__ == '__main__':
    dataset = ExplaGraphsBaselineDataset()

    # print(dataset.prompt)

    data = dataset[0]
    for k, v in data.items():
        print(f'{k}: {v}')

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
