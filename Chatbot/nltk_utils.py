import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

#  nltk.download('punkt')

def tokenize(sentence):
    """ Break down sentence into their words """
    return nltk.word_tokenize(sentence)
    
def stem(word):
    """ Steming words to their roots """
    stemmer = PorterStemmer()
    return stemmer.stem(word.lower())
    
def bag_of_words(tokenize_sentence, all_words):
    """
    sentence = ['I', 'like', 'potatoes']
    words = ['I', 'like', 'apples' , 'potatoes', 'bananas']
    bag = [1,1,0,1,0]
    """
    tokenize_sentence = [stem(w) for w in tokenize_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32) # we need to keep to float bc pytorch use floats
    for idx, w in enumerate(all_words):
        if w in tokenize_sentence:
            bag[idx] = 1.0
    return bag


if __name__ == "__main__":
    # test stemming and tokenization
    sentence = "I want to eat apples and organize them in my fridge"
    tokenized_words = tokenize(sentence)
    stemmed_words = [stem(word) for word in tokenized_words]
    #  print(stemmed_words)

    # test bag of words
    sentence = ['I', 'like', 'potatoes']
    words = ['I', 'like', 'apples' , 'potatoes', 'bananas']
    print(bag_of_words(sentence, words))

