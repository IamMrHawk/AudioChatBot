import numpy as np
import json
from utils import Utils
from model import NeuNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ChatDataset import ChatDataset


class TrainModel:
    def __init__(self):
        with open('intents.json', 'r') as f:
            self.intents = json.load(f)

        # variables
        self.all_words = []
        self.tags = []
        self.xy = []
        self.ignore_words = ['?', '!', '.', ',']
        self.x_train = []
        self.y_train = []
        self.dataset_creation()

        # HyperParameters
        self.batch_size = 8
        self.input_size = len(self.x_train[0])
        self.hidden_size = 8
        self.num_classes = len(self.tags)
        self.learning_rate = 0.001
        self.num_epochs = 1000

        self.dataset = ChatDataset(self.x_train, self.y_train)
        self.train_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = NeuNet(self.input_size, self.hidden_size, self.num_classes).to(self.device)
        self.train()
        self.save_model()

    def dataset_creation(self):
        for intent in self.intents['intents']:
            tag = intent['tag']
            self.tags.append(tag)
            for pattern in intent['patterns']:
                w = Utils.tokenize(pattern)
                self.all_words.extend(w)
                self.xy.append((w, intent['tag']))

        all_words = [Utils().stem(w) for w in self.all_words if w not in self.ignore_words]
        all_words = sorted(set(all_words))
        tags = sorted(set(self.tags))

        for (pattern_sentence, tag) in self.xy:
            bag = Utils().bag_of_words(pattern_sentence, all_words)
            self.x_train.append(bag)
            label = tags.index(tag)
            self.y_train.append(label)

        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)

    def train(self):
        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            for (words, labels) in self.train_loader:
                # Get data to Cuda if possible

                # data = data.type(torch.LongTensor)
                # targets = targets.type(torch.LongTensor)

                words = words.to(device=self.device)
                labels = labels.to(dtype=torch.long).to(device=self.device)

                # forward
                outputs = self.model(words)
                loss = criterion(outputs, labels)

                # backward
                optimizer.zero_grad()
                loss.backward()

                # gradiant decent or adam step
                optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'epoch [{epoch+1}/{self.num_epochs}], loss={loss.item():.4f}')

        print(f'Final Loss, loss={loss.item():.4f}')

    def save_model(self):
        data = {
            "model_state": self.model.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_classes": self.num_classes,
            "all_words": self.all_words,
            "tags": self.tags
        }
        file = "data.pth"
        torch.save(data, file)
        print(f'training complete. file saved to {file}')


if __name__ == "__main__":
    train = TrainModel()
