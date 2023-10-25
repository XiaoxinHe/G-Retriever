import pandas as pd
import torch
from torch.utils.data import Dataset


PATH = 'dataset/tape_bioasq_baseline'


class BioASQBaselineDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.data = pd.read_json(f'{PATH}/yesno.json', orient='records', lines=True)
        self.graph = None
        self.graph_type = 'Medical Knowledge Graph'
        self.prompt = "Given the medical knowledge graph, answer the question in Yes or No: "

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        data = self.data.iloc[index]

        question = f'Question: {data["sent"]}\n{self.prompt}'
        pyg_graph = torch.load(f'{PATH}/graph/{index}.pt')

        with open(f'{PATH}/text/{index}.txt') as f:
            desc = f.readlines()
            desc = ''.join(desc)
            desc = f'{self.graph_type}: \n{desc}\n'

        return {
            'id': index,
            'question': question,
            'label': data['exact_answer'],
            'full_label': data['ideal_answer'],
            'desc': desc,
            'graph': pyg_graph
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
    dataset = BioASQBaselineDataset()

    data = dataset[0]
    for k, v in data.items():
        print(f'{k}: {v}')

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
