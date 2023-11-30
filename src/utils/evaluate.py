import json
import pandas as pd
import re


def get_accuracy_gqa(eval_output, path):

    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for _, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

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
        for _, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        matches = re.findall(r"support|Support|Counter|counter", pred.strip())
        if len(matches) > 0 and matches[0].lower() == label:
            correct += 1

    return correct/len(df)


def get_accuracy_webqsp(eval_output, path):
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    with open(path, 'w')as f:
        for _, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    all_hit = []
    all_precision = []
    all_recall = []
    all_f1 = []

    for pred, label in zip(df.pred.tolist(), df.label.tolist()):
        try:
            pred = pred.split('[/s]')[0].strip().split('|')
            hit = re.findall(pred[0], label)
            all_hit.append(len(hit) > 0)

            label = label.split('|')
            matches = set(pred).intersection(set(label))
            precision = len(matches)/len(set(label))
            recall = len(matches)/len(set(pred))
            if recall + precision == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)

        except:
            print(f'Label: {label}')
            print(f'Pred: {pred}')
            print('------------------')
    hit = sum(all_hit)/len(all_hit)
    precision = sum(all_precision)/len(all_precision)
    recall = sum(all_recall)/len(all_recall)
    f1 = sum(all_f1)/len(all_f1)

    print(f'Hit: {hit:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1: {f1:.4f}')

    return hit


eval_funcs = {
    'expla_graphs': get_accuracy_expla_graphs,
    'gqa': get_accuracy_gqa,
    'gqa_baseline': get_accuracy_gqa,
    'webqsp': get_accuracy_webqsp,
    'webqsp_baseline': get_accuracy_webqsp,
}
