import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset
import json


class GQADataset(Dataset):
    def __init__(self):
        super().__init__()
        self.questions = pd.read_csv('dataset/tape_gqa/questions.csv')
        self.prompt = None
        self.graph = None
        self.graph_type = 'Scene Graph'

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.questions)

    def __getitem__(self, index):
        data = self.questions.iloc[index]

        # load the scene graph on the fly
        image_id = data['image_id']
        scene_graph_pyg = torch.load('dataset/tape_gqa/graph/{}.pt'.format(image_id))

        with open('dataset/tape_gqa/text/{}.txt'.format(image_id)) as f:
            desc = f.readlines()

        question = f'Question: {data["question"]}\nAnswer: '
        return {
            'id': data['q_id'],
            'image_id': data['image_id'],
            'question': question,
            'label': data['answer'],
            'full_label': data['full_answer'],
            'graph': scene_graph_pyg,
            'desc': desc,
        }

    def get_idx_split(self):

        # Load the saved indices
        with open('dataset/tape_gqa/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open('dataset/tape_gqa/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open('dataset/tape_gqa/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}


if __name__ == '__main__':
    dataset = GQADataset()

    data = dataset[0]
    for k, v in data.items():
        print(f'{k}: {v}')

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
