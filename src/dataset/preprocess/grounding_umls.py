from multiprocessing import Pool
import re
from tqdm import tqdm
import pandas as pd
import os
import json
import spacy

# !IMPORTANT
import scispacy
from scispacy.linking import EntityLinker


def load_umls_vocab(vocab_path):
    print("Loading UMLS vocab...")
    with open(vocab_path, "r", encoding="utf8") as fin:
        vocab = {l.split('\t')[0]: l.split('\t')[1].strip() for l in fin}
    return vocab


def load_entity_linker(threshold=0.90):
    print("Loading scispacy...")
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe("scispacy_linker",
                 config={
                     "resolve_abbreviations": True,
                     "linker_name": "umls",
                     "threshold": threshold})
    linker = nlp.get_pipe("scispacy_linker")
    print("Loaded scispacy.")
    return nlp, linker


def entity_linking_to_umls(sentence, nlp, linker):
    doc = nlp(sentence)
    entities = doc.ents
    all_entities_results = []
    for mm in range(len(entities)):
        entity_text = entities[mm].text
        entity_start = entities[mm].start
        entity_end = entities[mm].end
        all_linked_entities = entities[mm]._.kb_ents
        all_entity_results = []
        for ii in range(len(all_linked_entities)):
            curr_concept_id = all_linked_entities[ii][0]
            curr_score = all_linked_entities[ii][1]
            curr_scispacy_entity = linker.kb.cui_to_entity[all_linked_entities[ii][0]]
            curr_canonical_name = curr_scispacy_entity.canonical_name
            curr_TUIs = curr_scispacy_entity.types
            curr_entity_result = {"Canonical Name": curr_canonical_name, "Concept ID": curr_concept_id,
                                  "TUIs": curr_TUIs, "Score": curr_score}
            all_entity_results.append(curr_entity_result)
        curr_entities_result = {"text": entity_text, "start": entity_start, "end": entity_end,
                                "start_char": entities[mm].start_char, "end_char": entities[mm].end_char,
                                "linking_results": all_entity_results}
        all_entities_results.append(curr_entities_result)
    return all_entities_results


def ground_mentioned_concepts(nlp, linker, sent):
    ent_link_results = entity_linking_to_umls(sent, nlp, linker)
    mentioned_concepts = set()
    for ent_obj in ent_link_results:
        for ent_cand in ent_obj['linking_results']:
            CUI = ent_cand['Concept ID']
            if CUI in UMLS_VOCAB:
                mentioned_concepts.add(CUI)
    return list(mentioned_concepts)


def main():

    global UMLS_VOCAB
    vocab_path = 'dataset/umls/concept_names.txt'
    UMLS_VOCAB = load_umls_vocab(vocab_path)

    global nlp, linker
    nlp, linker = load_entity_linker()

    data = pd.read_csv('dataset/bioasq_all/yesno.csv')

    res = []
    for i, row in tqdm(data.iterrows(), total=len(data)):
        concepts = ground_mentioned_concepts(nlp, linker, row.question)
        if len(concepts) == 0:
            continue
        res.append(
            {'sent': row.question,
             'qc': concepts,
             'exact_answer': row.exact_answer,
             'ideal_answer': row.ideal_answer}
        )

    os.makedirs('dataset/tape_bioasq', exist_ok=True)
    output_path = 'dataset/tape_bioasq/yesno.json'
    df = pd.DataFrame(res)
    df.to_json(output_path, orient='records', lines=True)
    print(f'grounded concepts saved to {output_path}')
    print()


if __name__ == '__main__':
    main()
