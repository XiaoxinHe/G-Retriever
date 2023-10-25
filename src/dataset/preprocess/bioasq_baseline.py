import pandas as pd
import os


def preprocess():
    os.makedirs('dataset/tape_bioasq_baseline/text', exist_ok=True)

    df = pd.read_json('dataset/tape_bioasq/yesno.json', orient='records', lines=True)
    for i in (range(len(df))):
        with open(f'dataset/tape_bioasq/text/{i}.txt', 'r') as f:
            lines = f.readlines()
        lines = [line.replace(' info: <NodeEmb>;', '') for line in lines]
        with open(f'dataset/tape_bioasq_baseline/text/{i}.txt', 'w') as f:
            f.writelines(lines)


if __name__ == '__main__':
    preprocess()
