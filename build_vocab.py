"""
source: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/build_vocab.py
"""
import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(caption_path,threshold,vocab_path):
    vocab = build_vocab(json=caption_path, threshold=threshold)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    import config
    import os
    import argparse

    parser = argparse.ArgumentParser()  
    parser.add_argument('--train_json_path', type=str, default=config.TRAIN_JSON_PATH, help='path for train json')
    parser.add_argument('--vocab_path', type=str, default=config.VOCAB_PATH, help='path for vocabulary wrapper')
    parser.add_argument('--vocab_threshold', type=str, default=config.VOCAB_THRESHOLD, help='minimum number of words in caption corpus to be added to vocab')
    args = parser.parse_args(args=[])

    if not os.path.isdir('./data'):
        # create default directory for storing vocab if not available
        os.mkdir('./data')
    
    main(args.train_json_path,args.vocab_threshold,args.vocab_path)
    