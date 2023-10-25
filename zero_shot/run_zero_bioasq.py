import pandas as pd

import os
import concurrent.futures
import openai
import json


OPENAI_API_KEY = "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"
path = 'dataset/bioasq'


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


def load_bioasq():
    data = []
    with open(f'{path}/yesno.json') as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    concept2name = {}
    for line in open(f'dataset/umls/concept_names.txt'):
        c, n = line.strip().split('\t')
        concept2name[c] = n
    df['qc_name'] = df.qc.apply(lambda x: [concept2name[c] for c in x])
    # df.to_csv(f'{path}/yesno.csv', index=False,
    #           columns=['sent', 'qc', 'qc_name', 'exact_answer', 'ideal_answer'],)

    return df


def load_data():

    df = load_bioasq()

    prompt = 'Provide your answer in one word in the form of \"yes\" or \"no\", then provide your explanations in next lines.\n'

    os.makedirs(f'{path}/gpt_responses', exist_ok=True)
    queries = []
    fpaths = []
    for index, row in df.iterrows():
        q = f'Question: {row.sent}\n{prompt}'
        queries.append(q)
        fpaths.append(f'{path}/gpt_responses/' + str(index) + '.txt')
    return fpaths, queries, df.exact_answer.tolist()


def main():
    fpaths, queries, labels = load_data()

    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        res = executor.map(query_api, fpaths, queries, labels)


if __name__ == '__main__':
    main()
