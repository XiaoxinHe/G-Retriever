import json
import pandas as pd
import re

import glob


def get_accuracy_bioasq_chatgpt():

    files = glob.glob('dataset/bioasq/gpt_responses/*.txt')

    classes = ['Yes', 'No']
    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    for f in files:
        res = json.load(open(f))
        label = res['label']
        pred = res['choices'][0]['message']['content']
        matches = re.findall(classes_regex, pred)
        if len(matches) > 0 and matches[0].lower() == label.lower():
            correct += 1

    print(f'Accuracy: {correct/len(files)}')
    print(f'Total: {len(files)}')


def get_accuracy_bioasq(eval_output, path):
    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    classes = ['Yes', 'No', 'yes', 'no']
    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        matches = re.findall(classes_regex, pred)
        if len(matches) > 0 and matches[0].lower() == label.lower():
            correct += 1

    return correct/len(df)


def get_accuracy_cora(eval_output, path):

    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    classes = ['Case Based', 'Genetic Algorithms', 'Neural Networks',
               'Probabilistic Method', 'Reinforcement Learning', 'Rule Learning', 'Theory']
    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        matches = re.findall(classes_regex, pred)
        if len(matches) > 0 and matches[0] == label:
            correct += 1

    return correct/len(df)


def get_accuracy_pubmed(eval_output, path):

    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        if label in pred:
            correct += 1

    return correct/len(df)


def get_accuracy_citeseer(eval_output, path):

    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        if label in pred:
            correct += 1

    return correct/len(df)


def get_accuracy_arxiv(eval_output, path):

    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        matches = re.findall(r"cs\.[a-z]{2}", pred.strip())
        if len(matches) > 0 and label == matches[0]:
            correct += 1

    return correct/len(df)


def get_accuracy_gqa(eval_output, path):

    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        if label in pred:
            correct += 1
    return correct/len(df)


def get_accuracy_medgqa(eval_output, path):

    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    df.to_csv(path, index=False)

    # compute accuracy
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        if label in pred:
            correct += 1
    return correct/len(df)


def get_accuracy_expla_graphs(eval_output, path):

    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    with open(path, 'w')as f:
        for k, v in df.T.to_dict().items():
            f.write(json.dumps(v)+'\n')

    # compute accuracy
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        matches = re.findall(r"support|Support|Counter|counter", pred.strip())
        if len(matches) > 0 and matches[0].lower() == label:
            correct += 1

    return correct/len(df)


eval_funcs = {
    'cora': get_accuracy_cora,
    'citeseer': get_accuracy_citeseer,
    'pubmed': get_accuracy_pubmed,
    'arxiv': get_accuracy_arxiv,
    'gqa': get_accuracy_gqa,
    'gqa_baseline': get_accuracy_gqa,
    'medqa': get_accuracy_medgqa,
    'expla_graphs': get_accuracy_expla_graphs,
    'expla_graphs_baseline': get_accuracy_expla_graphs,
    'bioasq': get_accuracy_bioasq,
    'bioasq_baseline': get_accuracy_bioasq,
}
