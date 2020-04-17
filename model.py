import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

device = torch.device("cpu")

class BahdanauAttention(nn.Module):
    def __init__(self,embedding_dim,hidden_size):
        super().__init__()
        """
        embedding_dim = dimension of words must be same as output dim of pictures
        hidden_size = dim of hidden size of lstm/gru
        """
        self.w1 = nn.Linear(embedding_dim,embedding_dim)
        self.w2 = nn.Linear(hidden_size,embedding_dim)
        self.v = nn.Linear(embedding_dim,1)

    def forward(self,features,hidden):
        
        hidden_time = hidden.unsqueeze(1)
        attention_score = torch.tanh(self.w1(features)+self.w2(hidden_time))
        attention_weights = F.softmax(self.v(attention_score),dim=1)
        context_vector = features * attention_weights
        context_vector = torch.sum(context_vector,dim=1)

        return context_vector,attention_weights


    

class EncoderCNN(nn.Module):
    """
    Just pass in features from pickle files
    """
    def __init__(self,input_size,embedding_dim=256):
        """
        input_size = num of channels from cnn
        use fc to convert 512 x 49 -> embedding_dim x 49 
        Can think of it as compressing number of channels or increasing number of channels
        """
        super().__init__()
        self.fc = nn.Linear(input_size,embedding_dim)
        self.bn = nn.BatchNorm1d(49)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderRNN(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size) #this embeddings will be learned
        self.lstm = nn.LSTMCell(embed_size+embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        self.attention = BahdanauAttention(embed_size,hidden_size)
    
    def forward(self,images,captions,lengths):
        """Decode image feature vectors and generates captions."""
        batch_size = images.size(0)
        num_pixels = images.size(1)
        embeddings = self.embed(captions) # embed words
        lengths = torch.Tensor(lengths).long()
        decode_lengths = (lengths - 1).tolist()
        
        h,c = self.init_hidden(batch_size)

        #device here will us the nearest scope
        predictions = torch.zeros(batch_size, max(lengths), self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(lengths), num_pixels).to(device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(images[:batch_size_t],
                                                                h[:batch_size_t])
            h, c = self.lstm(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))
            

            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha.squeeze(2)

        return predictions,captions,lengths,alphas
    
    def init_hidden(self,batch_size):
        h = torch.zeros(batch_size,self.hidden_size)
        c = torch.zeros(batch_size,self.hidden_size)
        return h,c

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