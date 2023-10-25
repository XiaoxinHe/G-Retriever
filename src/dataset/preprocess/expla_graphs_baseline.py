import json
import re
import os
import torch
import pandas as pd
import numpy as np
import gensim
from tqdm import tqdm
from torch_geometric.data.data import Data

from src.dataset.preprocess.generate_split import generate_split

hidden_dim = 300

PATH = 'dataset/expla_graphs'
TAPE_PATH = 'dataset/tape_expla_graphs_baseline'


def to_text(input_string, index):

    # Use regular expressions to extract relationships
    relationships = re.findall(r'\((.*?)\)', input_string)

    triplets = []

    # Iterate through the extracted relationships and split them into subject, predicate, and object
    for relationship in relationships:
        parts = relationship.split('; ')
        if len(parts) == 3:
            subject, predicate, obj = parts
            triplets.append((subject, predicate, obj))

    graph = []
    node = {}
    edges = {}

    # Loop through the data and create the dictionary
    for triple in triplets:
        source, relation, target = triple

        if source not in node:
            node[source] = len(node)
        if target not in node:
            node[target] = len(node)
        source_index = node[source]
        dest_index = node[target]
        if source_index not in edges:
            edges[source_index] = []
        edges[source_index].append(tuple((relation, dest_index)))

    for i, node_desc in enumerate(node):
        if i in edges:
            graph.append(f'{i}: name: {node_desc}; relations: {edges[i]}\n'.replace('\'', ''))
        else:
            graph.append(f'{i}: name: {node_desc}\n')

    os.makedirs(f'{TAPE_PATH}/text', exist_ok=True)
    with open(f'{TAPE_PATH}/text/{index}.txt', 'w') as file:
        file.writelines(graph)


def encode_text_with_word2vec(text):
    words = text.split()  # Tokenize the text into words
    word_vectors = []

    for word in words:
        try:
            vector = word2vec[word]  # Get the Word2Vec vector for the word
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


def merge_train_dev():
    train = pd.read_csv(f'{PATH}/train.tsv', sep='\t', header=None)
    dev = pd.read_csv(f'{PATH}/dev.tsv', sep='\t', header=None)
    df = pd.concat([train, dev])
    df.columns = ['arg1', 'arg2', 'label', 'graph']

    os.makedirs(TAPE_PATH, exist_ok=True)
    df.to_csv(f'{TAPE_PATH}/train_dev.tsv', sep='\t', index=False)


def load_word2vec(path):
    print('Loading Google\'s pre-trained Word2Vec model...')
    global word2vec
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    print('Finish loading word2vec model!')
    return word2vec


if __name__ == '__main__':

    print("Preprocessing tape_expla_graphs_baseline dataset...")

    merge_train_dev()

    df = pd.read_csv(f'{TAPE_PATH}/train_dev.tsv', sep='\t')

    # save split
    generate_split(len(df), f'{TAPE_PATH}/split')

    # generate text description
    for index, row in tqdm(df.iterrows(), total=len(df)):
        to_text(row.graph, index)

    print("Done!")
