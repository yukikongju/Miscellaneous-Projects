#!/env/bin/python

#  from nltk.corpus import udhr
#  from nltk.corpus import gutenberg

import requests
import os
import nltk

from nltk.corpus import stopwords


def load_gutenberg_text():
    path = 'PDF-Searcher/gutenberg.txt'

    if not os.path.exists(path):
        url = 'https://www.gutenberg.org/files/2554/2554-0.txt'
        r = requests.get(url)
        text = r.content

        with open(path, 'w') as f:
            text = text.decode("utf-8")
            f.write(text)
    else: 
        with open(path, 'r') as f:
            text = f.read()
            
    return text

def remove_stop_words(text):
    partial_text = text[1500:10000]
    words = partial_text.split(' ')
    #  words.replace('\n', ' ').replace('\t', ' ')
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    return filtered_words



def main():
    text = load_gutenberg_text()
    filtered_text = remove_stop_words(text)
    print(filtered_text)
    pass
    

if __name__ == "__main__":
    main()
