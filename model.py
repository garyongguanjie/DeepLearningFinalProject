import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class BahdanauAttention(nn.Module):
    def __init__(self,image_dim,hidden_size):
        super().__init__()
        """
        image_dim = num of 'channels' from image encoder
        hidden_size = dim of hidden size of lstm/gru
        """
        self.w1 = nn.Linear(image_dim,image_dim)
        self.w2 = nn.Linear(hidden_size,image_dim)
        self.v = nn.Linear(image_dim,1)

    def forward(self,features,hidden):
        
        hidden_time = hidden.unsqueeze(1)
        attention_score = torch.tanh(self.w1(features)+self.w2(hidden_time))
        attention_weights = F.softmax(self.v(attention_score),dim=1)
        context_vector = features * attention_weights
        context_vector = torch.sum(context_vector,dim=1)

        return context_vector,attention_weights

class CNNfull(nn.Module):
    """
    passes in full image
    """
    def __init__(self,fine_tune=3):
        """
        input_size = num_channels from cnn
        fine_tune: num of blocks onwards of which we update params of resnet
        Eg if fine tune =3: only 3rd block onwards of resnet will have grad updated
        """
        super().__init__()
        model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2]) #chop off last two layers
        
        for params in self.model.parameters():
            params.requires_grad = False

        for children in list(self.model.children())[fine_tune:]:
            for params in children.parameters():
                params.requires_grad = True

    def forward(self,x):
        x = self.model(x) # bs x 2048 x 7 x 7
        bs,c,h,w = x.shape
        x = x.view(bs,-1,h*w) # bs x 2048 x 49
        x = x.permute(0,2,1) # bs x 49 x 2048
        return x

class EncoderCNN(nn.Module):
    """
    Just pass in features from pickle files
    """
    def __init__(self,input_size,image_dim):
        """
        input_size = num of channels from cnn
        use fc to convert 512 x 49 -> image_dim x 49 
        Can think of it as compressing number of channels or increasing number of channels
        """
        super().__init__()
        self.fc = nn.Linear(input_size,image_dim)
        self.bn = nn.BatchNorm1d(49)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = x.permute(0,2,1)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderRNN(nn.Module):

    def __init__(self, image_dim,embed_size, hidden_size, vocab_size):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size) #this embeddings will be learned
        
        # for learning start hidden states from images
        self.init_c = nn.Linear(image_dim,hidden_size)
        self.init_h = nn.Linear(image_dim,hidden_size)
        
        self.lstm = nn.LSTMCell(embed_size+image_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        self.attention = BahdanauAttention(image_dim,hidden_size)
    
    def forward(self,images,captions,lengths):
        """Decode image feature vectors and generates captions."""
        batch_size = images.size(0)
        num_pixels = images.size(1)
        embeddings = self.embed(captions) # bs,max_seq_length,embed_dimension
        lengths = torch.Tensor(lengths).long()
        decode_lengths = (lengths - 1).tolist()
        
        h,c = self.init_hidden(batch_size,images)

        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

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
        #alphas are the attention weights
        return predictions,captions,decode_lengths,alphas
    
    def init_hidden(self,batch_size,images):
        images = images.mean(dim=1)
        h = self.init_h(images)
        c = self.init_c(images)
        return h,c

    def inference(self, images,max_seq_length=30):
        """Decode image feature vectors and generates captions."""
        batch_size = images.size(0)
        num_pixels = images.size(1)
        
        h,c = self.init_hidden(batch_size,images)

        predictions = torch.zeros(batch_size,max_seq_length, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size,max_seq_length, num_pixels).to(device)
        lengths = torch.zeros(batch_size).to(device)

        embeddings = self.embed(torch.ones(batch_size).long().to(device))

        for t in range(max_seq_length):

            attention_weighted_encoding, alpha = self.attention(images,h)
            h, c = self.lstm(
                torch.cat([embeddings, attention_weighted_encoding], dim=1),
                (h, c))
            
            preds = self.fc(h)  # (batch_size_t, vocab_size)


            predictions[:, t, :] = preds
            preds = preds.argmax(dim=1)

            embeddings = self.embed(preds)
            lengths[(preds == 2) & (lengths==0)] = t + 1

            alphas[:, t, :] = alpha.squeeze(2)
        #alphas are the attention weights
        return predictions,lengths,alphas
