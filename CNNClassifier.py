import pandas as pd
import re
import torch.nn as nn
import torch
import numpy as np
from typing import List
import torch.nn.functional as F
from torch import optim

SEED = 42
torch.manual_seed(SEED)

df = pd.read_csv("data.txt", encoding="latin-1", sep="\013", header=None)
df = df[0].str.split(" ", n=1, expand=True)
df.rename({0: "Output", 1: "Sentence"}, axis="columns", inplace=True)

sentences = df["Sentence"]
labels = df["Output"]


def word_extraction(sentence):
    ignore = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

    words = re.sub("[^\w]", " ", sentence).split()
    cleaned_text = [w.lower() for w in words if w not in ignore]

    return cleaned_text


def clean_sentences(sentences):
    extracted_sentences = []
    for sent in sentences:
        extracted_sentences.append(word_extraction(sent))
    return extracted_sentences


def build_label_dict():
    """ Give each label a unique number using the training dataset to keep label numbering consistent """
    label_to_id = {}
    count = 0
    training_labels = df["Output"]
    for label in training_labels:
        label_exist = label_to_id.get(label) is not None
        if not label_exist:
            label_to_id[label] = count
            count += 1

    return label_to_id


def load_embeddings():
    embedding_dict = {}
    with open("glove.small.txt", "rt") as fi:
        full_content = fi.read().strip().split("\n")

    for i in full_content:
        temp_embeddings = i.split(" ")
        first_embedding_item = temp_embeddings[0].split("\t")
        word = first_embedding_item[0]
        first_embedding_digit = first_embedding_item[1]
        temp_embeddings[0] = first_embedding_digit

        embeddings = [float(val) for val in temp_embeddings]
        embedding_dict[word.lower()] = embeddings
        embedding_dict[word.lower()] = embeddings

    return embedding_dict


def create_embedding_matrix(embedding_dict, vocab_dict):
    # Let the first row 0 of the matrix be the padding
    embedding_length = 300

    first_row = [0] * embedding_length
    embedding_matrix = [first_row]
    for word, _id in vocab_dict.items():
        if embedding_dict.get(word):
            embedding_matrix.append(embedding_dict.get(word))

        else:
            embedding_matrix.append(embedding_dict.get("#unk#"))

    return np.array(embedding_matrix)


def model_accuracy(predict, y):
    true_predict = (predict == y).float()
    acc = true_predict.sum() / len(true_predict)
    return acc


def train(model, train_dataset, optimizer, criterion):
    total_loss = 0.0
    total_acc = 0.0

    for batch_number, data in train_dataset.items():
        padded_sentences = data["padded_sentences"]
        labels = data["labels_"]
        sentence_lengths = data["sentence_lengths"]

        optimizer.zero_grad()

        output = model(padded_sentences, sentence_lengths)
        print("outputs ", output.argmax(axis=1))
        print("labels ", labels)

        loss = criterion(output, labels)
        acc = model_accuracy(output.argmax(axis=1), labels)
        print("acc ", acc)

        loss.backward()

        # update the weights
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()

    avg_total_loss = total_loss / len(train_dataset.items())
    avg_total_acc = total_acc / len(train_dataset.items())

    return avg_total_loss, avg_total_acc


def evaluate(model, val_dataset, criterion):
    total_loss = 0.0
    total_acc = 0.0

    # deactivating dropout layers
    model.eval()

    # deactivates autograd
    with torch.no_grad():
        for batch_number, data in val_dataset.items():
            padded_sentences = data["padded_sentences"]
            labels = data["labels_"]
            sentence_lengths = data["sentence_lengths"]

            output = model(padded_sentences, sentence_lengths)
            # print("outputs ", output.argmax(axis=1))
            # print("labels ", labels)

            loss = criterion(output, labels)
            acc = model_accuracy(output.argmax(axis=1), labels)
            # print("acc ", acc)

            total_loss += loss.item()
            total_acc += acc.item()

    avg_total_loss = total_loss / len(val_dataset.items())
    avg_total_acc = total_acc / len(val_dataset.items())

    return avg_total_loss, avg_total_acc


def load_test_dataset():
    df_test = pd.read_csv("test.txt", encoding="latin-1", sep="\013", header=None)
    df_test = df_test[0].str.split(" ", n=1, expand=True)
    df_test.rename({0: "Output", 1: "Sentence"}, axis="columns", inplace=True)

    sentences = df_test["Sentence"]
    labels = df_test["Output"]

    # Pass in the unprocessed datasets sentences and labels
    cleaned_dataset = CleanData(sentences, labels)
    VOCAB_DICT = cleaned_dataset.vocab_dict

    # test and dev splitting can be performed on SENTENCE_AS_IDS and LABELS_AS_IDS using sklearn
    SENTENCE_AS_IDS = cleaned_dataset.sentences_as_id()
    LABELS_AS_IDS = cleaned_dataset.labels_as_id()
    BATCH_SIZE = 1

    preprocess_test = PreProcess(SENTENCE_AS_IDS, LABELS_AS_IDS, VOCAB_DICT, BATCH_SIZE)
    test_dataset = preprocess_test.create_batched_dataset()

    return test_dataset, VOCAB_DICT


def test_model(model):
    test_dataset, VOCAB_DICT = load_test_dataset()
    total_dataset = len(test_dataset)

    # load weights for best performing model
    path = 'saved_weights_cnn.pt'
    model.load_state_dict(torch.load(path))
    model.eval()

    reverse_vocab_dict = {val: key for key, val in VOCAB_DICT.items()}
    correctly_predicted = 0

    for _, data in test_dataset.items():
        padded_sentences = data["padded_sentences"]
        labels = data["labels_"]
        sentence_lengths = data["sentence_lengths"]

        for sentence, sentence_length, label in zip(padded_sentences, sentence_lengths, labels):
            sentence_ = sentence.tolist()
            sentence_length_ = int(sentence_length)
            label_ = int(label)

            output = model(torch.LongTensor([sentence_]), torch.LongTensor([sentence_length_])).argmax(axis=1)
            prediction = int(output)

            print("\nSENTENCE ==> ", " ".join([reverse_vocab_dict.get(_id) or "" for _id in sentence_]))
            print("PREDICTED LABEL ==> ", prediction)
            print("CORRECT LABEL ==> ", label_)

            if prediction == label_:
                correctly_predicted += 1

    print("\n CORRECTLY PREDICTED ", correctly_predicted)

    print("INCORRECTLY PREDICTED ", total_dataset - correctly_predicted)

    accuracy = (correctly_predicted / total_dataset) * 100
    print("ACCURACY ", accuracy)

    return accuracy


class CleanData:
    def __init__(self, dataset_sentence: List[str], dataset_labels: List[str]):
        self.embedding_dict = load_embeddings()
        self.labels_dict = build_label_dict()
        self.vocab_dict = self._get_vocab_dict
        self.dataset_sentence = dataset_sentence
        self.dataset_labels = dataset_labels

    def _get_clean_sentences(self):
        """Removes stop words etc from sentences in the dataset"""
        return clean_sentences(self.dataset_sentence)

    @property
    def _get_vocab_dict(self):
        # Give each vocab a unique number
        vocab_to_id = {}
        for index, (word, _) in enumerate(self.embedding_dict.items()):
            vocab_to_id[word] = index + 1
        return vocab_to_id

    def get_vocab_size(self):
        return len(self.vocab_dict.keys())

    def sentences_as_id(self) -> List[List[int]]:
        cleaned_sentences_ = self._get_clean_sentences()
        sentence_repr = []
        for sentence in cleaned_sentences_:
            sentence_repr.append([self.vocab_dict.get(word) or 0 for word in sentence])
        return sentence_repr

    def labels_as_id(self):
        labels_as_id = []
        for label_name in self.dataset_labels:
            label_id = self.labels_dict.get(label_name)
            labels_as_id.append(label_id)
        return labels_as_id

    def get_embedding_matrix(self):
        return create_embedding_matrix(self.embedding_dict, self.vocab_dict)


class PreProcess:
    def __init__(self, sentences_: List[List[int]], labels: List[int], vocab_dict: dict, batch_size: int):
        self.sentences_ = sentences_
        self.labels = labels
        self.vocab_dict = vocab_dict
        self.batch_size = batch_size
        self.batch_datasets = {}

    @property
    def _sentences_with_labels(self):
        return zip(self.sentences_, self.labels)

    @property
    def _sorted_sentences(self):
        """ Sort sentences in a batch in descending order based on the number of tokens """
        return sorted(self._sentences_with_labels, key=lambda x: len(x[0]), reverse=True)

    @property
    def _batch_dataset(self):
        data_set_size = len(self._sorted_sentences)
        remainder = (data_set_size % self.batch_size)
        divisible_dataset = data_set_size - remainder

        batches_ = int(divisible_dataset / self.batch_size)
        batches = [self.batch_size] * batches_
        if remainder > 0:
            batches.append(remainder)

        return batches

    def pad_sentence(self, batched_dataset: List):
        """
        Make sentences same length by padding with 0 and return
        the length of each sentence in the batch and it label
        """
        sentence_lengths = []
        padded_sentences = []
        labels_ = []

        max_sentence = len(batched_dataset[0][0])
        for sentence, label in batched_dataset:
            sentence_length = len(sentence)
            padding_length = max_sentence - sentence_length
            sentence_lengths.append(sentence_length)

            padding = [0] * padding_length
            padded_sentences.append(sentence + padding)
            labels_.append(label)

        return padded_sentences, labels_, sentence_lengths

    def create_batched_dataset(self):
        batches = self._batch_dataset
        index = 0
        for i, x in enumerate(batches):
            index += x
            from_ = i * batches[i - 1]
            to = index
            batched_dataset = self._sorted_sentences[from_:to]
            padded_sentences, labels_, sentence_lengths = self.pad_sentence(batched_dataset)

            self.batch_datasets[i + 1] = {
                "padded_sentences": torch.LongTensor(padded_sentences),
                "labels_": torch.LongTensor(labels_),
                "sentence_lengths": torch.LongTensor(sentence_lengths),
            }

        return self.batch_datasets


class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden, n_label, n_layers, embedding_matrix, random_init):
        super(CNNModel, self).__init__()
        filters = [2, 3, 4]
        dropout = 0.5
        n_filters = 100

        # embedding layer
        if not random_init:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32),
                                                          freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=n_filters,
                      kernel_size=fs,
                      padding=fs // 2)
            for fs in filters
        ])

        self.fc = nn.Linear(len(filters) * n_filters, n_label)

        self.dropout = nn.Dropout(dropout)

    def forward(self, sentences, sentence_lengths=None):
        # sentences = [batch size, sent len]

        embedded = self.embedding(sentences)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.permute(0, 2, 1)

        # embedded = [batch size, emb dim, sent len]

        conved = [F.relu(conv(embedded)) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)


if __name__ == "__main__":
    # Pass in the unprocessed datasets sentences and labels
    cleaned_dataset = CleanData(sentences, labels)
    VOCAB_DICT = cleaned_dataset.vocab_dict

    # test and dev splitting can be performed on SENTENCE_AS_IDS and LABELS_AS_IDS using sklearn
    SENTENCE_AS_IDS = cleaned_dataset.sentences_as_id()
    LABELS_AS_IDS = cleaned_dataset.labels_as_id()
    BATCH_SIZE = 100

    preprocess_train = PreProcess(SENTENCE_AS_IDS[200:5452], LABELS_AS_IDS[200:5452], VOCAB_DICT, BATCH_SIZE)
    train_dataset = preprocess_train.create_batched_dataset()

    preprocess_valid = PreProcess(SENTENCE_AS_IDS[0:100], LABELS_AS_IDS[0:100], VOCAB_DICT, BATCH_SIZE)
    valid_dataset = preprocess_valid.create_batched_dataset()

    # preprocess_test = PreProcess(SENTENCE_AS_IDS[100:300], LABELS_AS_IDS[100:300], VOCAB_DICT, 1)
    # test_dataset = preprocess_test.create_batched_dataset()

    EMBEDDING_MATRIX = cleaned_dataset.get_embedding_matrix()
    VOCAB_SIZE = cleaned_dataset.get_vocab_size()

    EMBEDDING_DIM = 300
    HIDDEN = 100
    NUM_LABEL = 50
    NUM_LAYERS = 2

    model = CNNModel(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDING_DIM, hidden=HIDDEN, n_label=NUM_LABEL,
                     n_layers=NUM_LAYERS, embedding_matrix=EMBEDDING_MATRIX, random_init=False)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(reduction='sum')

    epochs = 10
    best_valid_loss = float('inf')

    for epoch in range(epochs):

        # train the model
        train_loss, train_acc = train(model, train_dataset, optimizer, criterion)

        # evaluate the model
        valid_loss, valid_acc = evaluate(model, valid_dataset, criterion)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_weights_cnn.pt')

        print(f'\tTrain Loss on epoch {epoch}: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss on epoch {epoch}: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    print(test_model(model))
