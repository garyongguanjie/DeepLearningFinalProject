import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

class BahdanauAttention(nn.Module):
    def __init__(self,image_dim,hidden_size):
        super().__init__()
        """
        image_dim = num of 'channels' from image encoder
        hidden_size = dim of hidden size of lstm/gru
        """
        self.w1 = nn.Linear(image_dim,image_dim)
        self.w2 = nn.Linear(hidden_size,image_dim)
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        self.v = nn.Linear(image_dim,1)

    def forward(self,features,hidden):
        
        hidden_time = hidden.unsqueeze(1)
        
        scoring = self.w1(features)+self.w2(hidden_time)
        for i,dropout in enumerate(self.dropouts):
            if i == 0:
                out = dropout(scoring)
            else:
                out += dropout(scoring)

        out /= len(self.dropouts)

        attention_score = torch.tanh(out)
        attention_weights = F.softmax(self.v(attention_score),dim=1)
        context_vector = features * attention_weights
        context_vector = torch.sum(context_vector,dim=1)

        return context_vector,attention_weights

class CNNfull(nn.Module):
    """
    passes in full image
    """
    def __init__(self,pretrained=True,fine_tune=3):
        """
        input_size = num_channels from cnn
        fine_tune: num of blocks onwards of which we update params of resnet
        Eg if fine tune =3: only 3rd block onwards of resnet will have grad updated
        """
        super().__init__()
        if pretrained:
            model = models.resnet50(pretrained=True)
        else:
            model = models.resnet50()

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

class DecoderRNN(nn.Module):

    def __init__(self, image_dim,embed_size, hidden_size, vocab_size,device=None):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        if device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        else:
            self.device = device
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size) #this embeddings will be learned
        self.embed_size = embed_size
        
        # for learning start hidden states from images
        self.init_c = nn.Linear(image_dim,hidden_size)
        self.init_h = nn.Linear(image_dim,hidden_size)
        
        self.lstm = nn.LSTMCell(embed_size+image_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        self.attention = BahdanauAttention(image_dim,hidden_size)
        
    def load_embeddings(self,embeddings):
        # set freeze = false as some words not in pretrained
        self.embed.from_pretrained(embeddings,freeze=False)

    def forward(self,images,captions,lengths):
        """Decode image feature vectors and generates captions."""
        device = self.device
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
            
            #multisample dropout for faster convergence
            for i,dropout in enumerate(self.dropouts):
                if i == 0:
                    out = dropout(h)
                else:
                    out += dropout(h)

            out /= len(self.dropouts)

            preds = self.fc(out)  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha.squeeze(2)
        #alphas are the attention weights
        return predictions,captions,decode_lengths,alphas
    
    def init_hidden(self,batch_size,images):
        images = images.mean(dim=1)
        h = self.init_h(images)
        c = self.init_c(images)
        return h,c

    def inference(self, images,max_seq_length=30, beam_width=1):
        """Decode image feature vectors and generates captions."""
        assert type(beam_width) == int
        assert beam_width >= 1
        assert beam_width < self.vocab_size
        # set max beam search width to reduce long computation times
        assert beam_width <= 10

        device = self.device
        batch_size = images.size(0)
        num_pixels = images.size(1)

        if beam_width == 1:
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
        
        elif beam_width > 1:
            # need to save hidden/cell states of LSTMs with different paths
            hidden_dict = {}
            cell_dict = {}
            # dictionary saves paths to get best path
            # dictionary format is {(timestep, node_number): (timestep-1, node_number)...}, where dict value leads to the dict key in a directed graph
            path_dict = defaultdict(lambda: [])
            pred_indices = torch.zeros(batch_size, beam_width, max_seq_length).to(device)
            pred_values = torch.zeros(batch_size, beam_width).to(device)
            alphas = torch.zeros(batch_size, beam_width, max_seq_length, num_pixels).to(device)
            lengths = torch.zeros(batch_size).to(device)

            embeddings = torch.ones(batch_size, beam_width, self.embed_size).float().to(device)

            # stop when <end> (to lower computational time) or when max_seq_length is reached
            for t in range(max_seq_length):
                if t == 0:
                    # initialise hidden, cell states and embeddings at the start
                    initial_embed = self.embed(torch.ones(batch_size).long().to(device))
                    h, c = self.init_hidden(batch_size, images)
                    attention_weighted_encoding, alpha = self.attention(images, h)
                    h, c = self.lstm(
                        torch.cat([initial_embed, attention_weighted_encoding], dim=1),
                        (h, c))

                    preds = self.fc(h)  # (batch_size_t, vocab_size)
                    preds = torch.exp(preds)
                    preds = torch.div(preds, torch.sum(preds, dim=1))
                    top_preds = torch.topk(preds, k=beam_width)
                    # 1) probabilities --> only need values from 1 timestep before
                    pred_values = top_preds.values # (batch_size, beam_width)
                    # 2) indices
                    pred_indices[:,:,t] = top_preds.indices # ()

                    for bw in range(beam_width):
                        # 3) hidden and cell states --> only need values from 1 timestep before
                        hidden_dict[bw] = h
                        cell_dict[bw] = c
                        # 4) path
                        path_dict[(t,bw)] = (-1,-1)
                        # 5) embeddings --> only need values from 1 timestep before
                        embeddings[:, bw] = self.embed(pred_indices[:,bw,t].long().to(device))
                        # 6) alphas
                        alphas[:, bw, t, :] = alpha.squeeze(2)
                    
                elif t >= 1:
                    # calculate predictons
                    temp_preds = torch.zeros(batch_size, beam_width, self.vocab_size).to(device)
                    temp_hidden_states = {}
                    temp_cell_states = {}
                    temp_alphas = {}

                    for bw in range(beam_width):
                        attention_weighted_encoding, temp_alphas[bw] = self.attention(images, hidden_dict[bw])
                        temp_hidden_states[bw], temp_cell_states[bw] = self.lstm(
                            torch.cat([embeddings[:, bw], attention_weighted_encoding], dim=1),
                            (hidden_dict[bw], cell_dict[bw]))
                        
                        preds = self.fc(temp_hidden_states[bw])  # (batch_size_t, vocab_size)
                        # TODO: increase numerical stability
                        preds = torch.exp(preds)
                        preds = torch.div(preds, torch.sum(preds,dim=1))
                        preds = torch.mul(pred_values[:,bw], preds)
                        temp_preds[:,bw,:] = preds

                    # get top predictions
                    top_preds = torch.topk(temp_preds, k=beam_width) # (batch_size, beam_width, beam_width)
                    top_preds_values_expanded = top_preds.values.reshape(batch_size,beam_width * beam_width) # (change to batch_size, beam_width * beam_width to run topk again, since topk only works with one dimension)
                    top_preds_indices_expanded = top_preds.indices.reshape(batch_size,beam_width * beam_width) # (as above)
                    top_preds_final_expanded = torch.topk(top_preds_values_expanded, k=beam_width)

                    # calculate and store current timestep's hidden/cell states, probabilities, indices, path, embeddings and alphas
                    for top_value_list in top_preds_final_expanded.indices:
                        for bw in range(len(top_value_list)): # length of top_value_list is the same as beam_width
                            node = int(np.ceil((top_value_list[bw].item() + 1) / beam_width) - 1)
                            # 1) probabilities --> only need values from 1 timestep before
                            pred_values[:,bw] = top_preds_values_expanded[0,top_value_list[bw].item()] # HERE, WE ASSUME THERE IS JUST BATCH_SIZE OF 1 FOR SIMPLICITY
                            # 2) indices
                            pred_indices[:,bw,t] = top_preds_indices_expanded[:,top_value_list[bw].item()]
                            # 3) hidden and cell states --> save only the chosen ones --> only need values from 1 timestep before
                            hidden_dict[bw] = temp_hidden_states[node]
                            cell_dict[bw] = temp_cell_states[node]
                            # 4) paths
                            path_dict[(t, bw)] = (t-1, node)
                            # 5) embeddings --> only need values from 1 timestep before
                            embeddings[:, bw] = self.embed(pred_indices[:,bw,t].long().to(device))
                            # 6) alphas
                            alphas[:, bw, t, :] = temp_alphas[node].squeeze(2)
                        break

                # check for <end> token
                current_indices = pred_indices[0,:,t].tolist() # HERE, WE ASSUME THERE IS JUST BATCH_SIZE OF 1 FOR SIMPLICITY
                if 2 in current_indices:
                    lengths[0] = t + 1 # HERE, WE ASSUME THERE IS JUST BATCH_SIZE OF 1 FOR SIMPLICITY
                    end_node = (t, current_indices.index(2))
                    break

            # check if <end> token was not reached
            if int(lengths[0].item()) == 0: # HERE, WE ASSUME THERE IS JUST BATCH_SIZE OF 1 FOR SIMPLICITY
                lengths[0] = max_seq_length # HERE, WE ASSUME THERE IS JUST BATCH_SIZE OF 1 FOR SIMPLICITY
                end_node = (max_seq_length - 1, np.argmax(pred_indices[0,:,t].tolist()))

            # Crawl backwards through path_dict to get best path
            predictions = torch.zeros(batch_size,max_seq_length, self.vocab_size).to(device)
            final_alphas = torch.zeros(batch_size,max_seq_length, num_pixels).to(device)
            path = []
            path.append(end_node)
            node = end_node
            for t in range(node[0], -1, -1):
                node = path_dict[node]
                path.append(node)
            # reverse path
            path = path[::-1]
            # get rid of first node (-1,-1)
            path = path[1:]
            # Standardise return Tensors with greedy search algorithm's results for easier post-processing
            for p in range(len(path)):
                idx = pred_indices[:, path[p][1], path[p][0]]
                predictions[:, p, int(idx.item())] = 1
                final_alphas[:, p, :] = alphas[:, path[p][1], p, :]

            alphas = final_alphas

        #alphas are the attention weights
        return predictions, lengths, alphas
