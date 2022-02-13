import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import tokenize, bag_of_words, stem
from model import NeuralNet


class Model(object):

    def __init__(self):
        # set intents
        self._set_intents_()

        # define hyperparameters
        self.num_epochs = 100
        self.batch_size = 8
        self.learning_rate = 0.001
        self.input_size = len(self.x_train[0])
        self.hidden_size = 8
        self.output_size = len(self.tags)
        
        # define model, loss and optimizer functions
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NeuralNet(self.input_size, self.hidden_size, self.output_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # define dataset and train_loader
        dataset = ChatDataset(self.x_train, self.y_train)
        self.train_loader = DataLoader(dataset = dataset, batch_size = self.batch_size, 
                shuffle=True, num_workers=0)


    def train_and_save_model(self):
        for epoch in range(self.num_epochs):
            for (words, labels) in self.train_loader:
                words = words.to(self.device)
                labels = labels.to(dtype=torch.long).to(self.device)

                # forward pass
                outputs = self.model(words)
                loss = self.criterion(outputs, labels)

                # backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')

        print(f'Final Loss: {loss.item():.4f}')

        self._save_model_()

    def _save_model_(self):
        data = {
            "model_state": self.model.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "all_words": self.all_words,
            "tags": self.tags
        }

        FILE = "Chatbot/data.pth"
        torch.save(data, FILE)

        print(f'training complete. file saved to {FILE}')

    def _set_intents_(self):
        # read intents file
        with open('Chatbot/intents.json', 'r') as f:
            intents = json.load(f)

        all_words = []
        tags = []
        xy = []

        for intent in intents['intents']:
            tag = intent['tag']
            tags.append(tag)
            for pattern in intent['patterns']:
                w = tokenize(pattern)
                all_words.extend(w)
                xy.append((w,tag))

        # remove punctations and unwanted words 
        punctuations = ['?', '!', '.', ',']
        all_words = [stem(w) for w in all_words]
        all_words = sorted(set(all_words))
        tags = sorted(set(tags))

        # create bag of words
        x_train = []
        y_train = []
        for (pattern_sentence, tag) in xy:
            bag = bag_of_words(pattern_sentence, all_words)
            x_train.append(bag)

            label = tags.index(tag)
            y_train.append(label)

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # save as class parameters
        self.x_train = x_train
        self.y_train = y_train
        self.all_words = all_words
        self.tags = tags
        
        # sanity check: verify that input and output are the right size
        #  print(input_size, len(all_words))
        #  print(output_size, len(tags))


class ChatDataset(Dataset):

    def __init__(self, x_train, y_train):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


    def __len__(self):
        return self.n_samples

if __name__ == "__main__":
    model = Model()
    model.train_and_save_model()

