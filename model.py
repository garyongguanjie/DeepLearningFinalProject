import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class BahdanauAttention(nn.Module):
    def __init__(self,units,embedding_dim):
        super().__init__()
        self.w1 = nn.Linear(embedding_dim,units)
        self.w2 = nn.Linear(embedding_dim,units)
        self.v = nn.Linear(embedding_dim,1)
    

class EncoderCNN(nn.Module):
    """
    Just pass in features from pickle files
    """
    def __init__(self,input_size,embedding_dim=256):
        """
        input_size = num of channels from cnn
        use fc to convert 512 x 49 -> 49 x embedding_dim
        """
        super().__init__()
        self.fc = nn.Linear(input_size,embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.relu = nn.Relu(inplace=True)
    def forward(self,x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DecoderRNN(nn.Module):
    """
    source:https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/model.py
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids