import random
import json
import numpy as np
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize 


class ChatBot:

    def __init__(self, intents, FILE):
        self.bot_name = "Bot"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.intents = intents
        self._read_data_(FILE)
        self._load_model_()

    def _read_data_(self, FILE):
        data = torch.load(FILE)

        self.input_size = data['input_size']
        self.hidden_size = data['hidden_size']
        self.output_size = data['output_size']
        self.all_words = data['all_words']
        self.tags = data['tags']
        self.model_state = data['model_state']

    def _load_model_(self):
        self.model = NeuralNet(self.input_size, self.hidden_size, self.output_size)
        self.model.load_state_dict(self.model_state)
        self.model.eval()

    def chat(self):
        print(f"Hi I am {self.bot_name}. How may I help you? (type 'quit' to exit)")
        while True: 
            sentence = input("You: ")
            if sentence == "quit":
                break

            response = self._get_bot_answer_(sentence)
            print(response)

        
    def _get_bot_answer_(self, sentence):
        sentence = tokenize(sentence)
        x = bag_of_words(sentence, self.all_words)
        x = x.reshape(1, x.shape[0])
        x = torch.from_numpy(x).to(self.device)

        output = self.model(x)
        _, predicted = torch.max(output, dim=1)

        tag = self.tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.55:
            for intent in self.intents['intents']:
                if tag == intent['tag']:
                    return f"{self.bot_name}: {random.choice(intent['responses'])}"
        else: 
            return f"{self.bot_name}: I did not understand..."

        
if __name__ == "__main__":
    # read intents file
    with open('Chatbot/intents.json', 'r') as f:
        intents = json.load(f)
    FILE = "Chatbot/data.pth"

    chatbot = ChatBot(intents, FILE)
    chatbot.chat()
    
