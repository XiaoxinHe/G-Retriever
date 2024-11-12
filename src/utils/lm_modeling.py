from tqdm import tqdm
import gensim
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np

pretrained_repo = 'sentence-transformers/all-roberta-large-v1'
batch_size = 1024  # Adjust the batch size as needed


# replace with the path to the word2vec file
word2vec_hidden_dim = 300
word2vec_path = 'word2vec/GoogleNews-vectors-negative300.bin.gz'


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_ids=None, attention_mask=None):
        super().__init__()
        self.data = {
            "input_ids": input_ids,
            "att_mask": attention_mask,
        }

    def __len__(self):
        return self.data["input_ids"].size(0)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        batch_data = dict()
        for key in self.data.keys():
            if self.data[key] is not None:
                batch_data[key] = self.data[key][index]
        return batch_data


class Sentence_Transformer(nn.Module):

    def __init__(self, pretrained_repo):
        super(Sentence_Transformer, self).__init__()
        print(f"inherit model weights from {pretrained_repo}")
        self.bert_model = AutoModel.from_pretrained(pretrained_repo)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        data_type = token_embeddings.dtype
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(data_type)
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, att_mask):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        sentence_embeddings = self.mean_pooling(bert_out, att_mask)

        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


def load_word2vec():
    print(f'Loading Google\'s pre-trained Word2Vec model from {word2vec_path}...')
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model, None, device


def text2embedding_word2vec(model, tokenizer, device, text):
    if type(text) is list:
        text_vector = torch.stack([text2embedding_word2vec(model, tokenizer, device, t) for t in text])
        return text_vector

    words = text.split()  # Tokenize the text into words
    word_vectors = []

    for word in words:
        try:
            vector = model[word]  # Get the Word2Vec vector for the word
            word_vectors.append(vector)
        except KeyError:
            # Handle the case where the word is not in the vocabulary
            pass

    if word_vectors:
        # Calculate the mean of word vectors to represent the text
        text_vector = sum(word_vectors) / len(word_vectors)
    else:
        # Handle the case where no word vectors were found
        text_vector = np.zeros(word2vec_hidden_dim)

    return torch.Tensor(text_vector)


def load_sbert():

    model = Sentence_Transformer(pretrained_repo)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_repo)

    # data parallel
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def sber_text2embedding(model, tokenizer, device, text):
    if len(text) == 0:
        return torch.zeros((0, 1024))

    encoding = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    dataset = Dataset(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask)

    # DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Placeholder for storing the embeddings
    all_embeddings = []

    # Iterate through batches
    with torch.no_grad():

        for batch in dataloader:
            # Move batch to the appropriate device
            batch = {key: value.to(device) for key, value in batch.items()}

            # Forward pass
            embeddings = model(input_ids=batch["input_ids"], att_mask=batch["att_mask"])

            # Append the embeddings to the list
            all_embeddings.append(embeddings)

    # Concatenate the embeddings from all batches
    all_embeddings = torch.cat(all_embeddings, dim=0).cpu()

    return all_embeddings


def load_contriever():
    print('Loading contriever model...')
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    model = AutoModel.from_pretrained('facebook/contriever')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    model.to(device)
    model.eval()
    return model, tokenizer, device


def contriever_text2embedding(model, tokenizer, device, text):

    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    try:
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        dataset = Dataset(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["att_mask"])
                embeddings = mean_pooling(outputs[0], batch['att_mask'])
                all_embeddings.append(embeddings)
            all_embeddings = torch.cat(all_embeddings, dim=0).cpu()
    except:
        all_embeddings = torch.zeros((0, 1024))

    return all_embeddings


load_model = {
    'sbert': load_sbert,
    'contriever': load_contriever,
    'word2vec': load_word2vec,

}


load_text2embedding = {
    'sbert': sber_text2embedding,
    'contriever': contriever_text2embedding,
    'word2vec': text2embedding_word2vec,
}
