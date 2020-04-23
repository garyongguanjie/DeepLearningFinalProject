import numpy as np
import pickle
import torch
from build_vocab import Vocabulary

def read_glove(file_name):
    embeddings_dict = {}
    with open(file_name,'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:],"float32")
            embeddings_dict[word] = vector
    return embeddings_dict

if __name__ == "__main__":
    import config
    with open(config.VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    embeddings_dict = read_glove('./data/glove.6B.300d.txt')
    embeddings = np.zeros((vocab_size,300),dtype=np.float32)

    counter = 0
    for i in range(vocab_size):
        word = vocab.idx2word[i]
        if word in embeddings_dict:
            counter += 1
            embeddings[i] = embeddings_dict[word]
        else:
            #loc -> mean of glove embeddings
            # scale -> std of glove embeddings
            embeddings[i] = np.random.normal(loc=-0.003905,scale=0.38177,size=(300,))
    print(f"number of words in glove are {counter}/{vocab_size}")
    torch_embeddings = torch.from_numpy(embeddings)
    torch.save(torch_embeddings,config.GLOVE_EMBED_PATH)

    

