import pandas as pd

import os
import concurrent.futures
import openai
import json


OPENAI_API_KEY = "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"


def query_api(fpath, prompt, label):
    isExist = os.path.exists(fpath)
    if isExist:
        return
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300,
        api_key=OPENAI_API_KEY)

    response = response.to_dict()
    response['prompt'] = prompt
    response['label'] = label

    outputs_file = open(fpath, "w")
    outputs_file.write(json.dumps(response, indent=4))


def load_data():

    df = pd.read_csv('dataset/expla_graphs/dev.tsv', sep='\t', header=None)
    df.columns = ['arg1', 'arg2', 'label', 'graph']

    prompt = 'Given the relation graph, argument 1 and argument 2 support or counter each other? Provide your answer in one word in the form of \"support\" or \"counter\", then provide your reasoning in next lines.\n'

    queries = []
    fpaths = []
    for index, row in df.iterrows():
        temp = f'Argument 1: {row.arg1}\nArgument 2: {row.arg2}\nRelation graph: {row.graph}\n\n{prompt}\n'
        queries.append(temp)
        fpaths.append('dataset/expla_graphs/gpt_responses/' + str(index) + '.txt')
    return fpaths, queries, df.label.tolist()


def main():
    fpaths, queries, labels = load_data()

    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        res = executor.map(query_api, fpaths, queries, labels)


if __name__ == '__main__':
    main()
