import argparse
import pandas as pd
import torch
import re
import warnings

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn import preprocessing

# Arguments that canbe used for the
parser = argparse.ArgumentParser(description='Program to train a model \
                                                for acronym disambiguation')
parser.add_argument("--test_set", default='coding_test_toy_set_acronyms.csv')
parser.add_argument("--train_set", default='coding_train_set_acronyms.csv')
parser.add_argument("--test_sentences", default='coding_test_sentence_data.txt')
parser.add_argument("--num_classes", type=int, default=6)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument("--device", default='cpu')
args = parser.parse_args()

class AcronymsDataset(Dataset):
    def __init__(self, csv_file, encoder):
        df = pd.read_csv(csv_file, sep="|")

        acronyms = df['acronym'].tolist()
        self.expansions = df['expansion'].tolist()
        self.samples = df['sample'].tolist()

        # Create indexes for each expansion label
        encoder.fit(self.expansions)
        self.expansions = encoder.transform(self.expansions)

        # Apoend acronym to sample
        self.samples = [acronyms[i] + " " + self.samples[i] for i in range(len(acronyms))]

    def __len__(self):
        return len(self.expansions)

    def __getitem__(self, idx):
        return self.samples[idx], self.expansions[idx]


class LSTMDisambiguation(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=300, hidden_dim=300):
        super(LSTMDisambiguation, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        embedding = self.embedding(x)
        padded_embedding = pad_sequence(embedding, batch_first=True)
        packed_output, _ = self.lstm(padded_embedding)

        output = self.fc(packed_output)

        return output


def train(model, data_loader, vocab, optimizer, loss_function, num_epochs, device):
    
    model = model.to(device)
    model.train()
    
    # train model for n epochs
    for n in range(num_epochs):
        for samples, expansions in data_loader:

            samples = process_input(vocab, samples)
            samples = torch.tensor(samples).to(device)

            output = model(samples)
            expansions = create_one_hot(expansions)

            loss = loss_function(output, expansions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return

def test(model, data_loader, vocab, loss_function, encoder, device):
    model.eval()
    with torch.no_grad():
        for samples, expansions in data_loader:

            encoded_samples = process_input(vocab, samples)
            encoded_samples = torch.tensor(encoded_samples).to(device)
            output = model(encoded_samples)
            predictions = torch.argmax(torch.argmax(output, dim=1))

            one_hot_expansions = create_one_hot(expansions)
            loss = loss_function(output, one_hot_expansions)

            print("acronym: " + samples[0].split(" ")[0])
            print("predicted expansion: " + encoder.inverse_transform(predictions))
            print("true expansion:      "  \ 
                    + encoder.inverse_transform(expansions).tostring() + "\n")

    return

# Get the vocabulary for this model
def get_vocab():
    chars="(),.:?;"
    with open(args.test_sentences, "r") as f:
        data = f.read().lower()

        # Remove characters from the dataset
        for char in chars:
            data = data.replace(char, "")
        
        # Split the data into words and remove empty entries
        data = re.split(" |\n", data)
        data = filter(lambda a : a != '', data)
        data = set(data)

        # Add Unknown for unused words
        data.add("UNK")

        return list(data)

# Used to proces to encode the input data for the model
def process_input(vocab, data, pad=0):
    chars="(),.:?;"
    processed_sentences = []

    for sentence in data:
        sentence = sentence.lower()
        for char in chars:
            sentence = sentence.replace(char, "")
        
        sentence = sentence.split(" ")
        new_sentence = []

        # Get the corresponding indification number for each word
        for word in sentence:
            if word in vocab:
                new_sentence.append(vocab.index(word))
            else:
                new_sentence.append(vocab.index("UNK"))
        
        # Add padding to make sure all sentences are the same length
        if len(new_sentence) > args.max_length:
            new_sentence = new_sentence[:args.max_length]
        else:
            new_sentence = new_sentence + ([pad] * (args.max_length - len(new_sentence)))
        
        processed_sentences.append(new_sentence)

    return processed_sentences

# Create a one hot vector from the labels
def create_one_hot(expansions):
    one_hots = []
    for expansion in expansions:
        one_hot = [0] * args.num_classes
        one_hot[expansion] = 1
        one_hots.append(one_hot)
    
    return torch.tensor(one_hots)


def main():
    vocab = get_vocab()
    vocab_size = len(vocab)

    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    le = preprocessing.LabelEncoder()

    train_dataset = AcronymsDataset(args.train_set, le)
    test_dataset = AcronymsDataset(args.test_set, le)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset)
    model = LSTMDisambiguation(vocab_size, args.num_classes)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, vocab, optimizer, loss_function, args.num_epochs, args.device)
    test(model, test_loader, vocab, loss_function, le, args.device)

if __name__ == "__main__":
    main()