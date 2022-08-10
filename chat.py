import json
import random
from utils import Utils
from model import NeuNet
import torch


class ChatBot:
    def __init__(self):
        with open('intents.json', 'r') as f:
            self.intents = json.load(f)
        file = 'data.pth'
        data = torch.load(file)

        self.input_size = data["input_size"]
        self.hidden_size = data["hidden_size"]
        self.num_classes = data["num_classes"]
        self.all_words = data['all_words']
        self.tags = data['tags']
        self.model_state = data['model_state']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = NeuNet(self.input_size, self.hidden_size, self.num_classes).to(self.device)
        self.model.load_state_dict(self.model_state)
        self.model.eval()

        self.bot = 'Jarvis'
        print('### This is a ChatBot program ###')
        print('please chat to interact and type "exit" to terminate program')
        self.main()

    def main(self):
        while True:
            sentence = input('Me : ')
            if sentence == 'exit':
                print('Thank you!! \n Terminating ChatBot')
                break
            sentence = Utils.tokenize(sentence)
            x = Utils().bag_of_words(tokenized_sentence=sentence, words=self.all_words)
            x = x.reshape(1, x.shape[0])
            x = torch.from_numpy(x).to(self.device)

            output = self.model(x)
            _, predicted = torch.max(output, dim=1)
            tag = self.tags[predicted.item()]

            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]

            if prob.item() > 0.75:
                for intent in self.intents['intents']:
                    if tag == intent['tag']:
                        print(f'{self.bot} : {random.choice(intent["responses"])}')
            else:
                print(f'{self.bot} : not understandable')

if __name__ == "__main__":
    chat_bot = ChatBot()
