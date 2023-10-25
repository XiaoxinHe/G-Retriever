from tqdm import tqdm
import pandas as pd
import os


def preprocess():
    os.makedirs('dataset/tape_gqa_baseline/text', exist_ok=True)

    df = pd.read_csv('dataset/tape_gqa/questions.csv')

    for index, row in tqdm(df.iterrows(), total=len(df)):
        image_id = row['image_id']
        with open(f'dataset/tape_gqa/text/{image_id}.txt', 'r') as f:
            lines = f.readlines()
        lines = [line.replace(' info: <NodeEmb>;', '') for line in lines]
        with open(f'dataset/tape_gqa_baseline/text/{image_id}.txt', 'w') as f:
            f.writelines(lines)


if __name__ == '__main__':
    preprocess()
