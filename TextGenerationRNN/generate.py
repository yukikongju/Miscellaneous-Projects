import tensorflow as tf
import numpy as np
import os 
import time

from tf.keras.layers import StringLookup


class Generator(object):

    def __init__(self, text):
        self.text = text
        self.unique_char = self._get_unique_characters()

    def _vectorize_text(self):
        pass

    def _get_unique_characters(self):
        return sorted(set(text))
        

def main():
    path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    # get shakespear text
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    print(f'Length of text: {len(text)} characters')

    # 

if __name__ == "__main__":
    main()

