import pickle
from data_loader import get_loader,CocoDataset, get_loader_unique
from build_vocab import Vocabulary
from torchvision import models
import torch
from torch import nn,optim
from torchvision import transforms
from model import DecoderRNN,CNNfull
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
from tqdm._utils import _term_move_up
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import argparse
from torchtext.data.metrics import bleu_score
from nltk.translate import meteor_score
from MyCocoDataset import MyCocoCaptions
import nltk
nltk.download('wordnet')
import sklearn

# Word Analogy: x1 = y1 - y2 + x2 E.g. boy = man - woman + girl  
def cosine_similarity_analogy(decoder,vocab,y1,y2,x2,topk=3):
    # Vocab indices
    vocab_word2idx = vocab.word2idx
    vocab_idx2word = vocab.idx2word
    
    # Extract embedding matrix from decoder
    embedding_matrix = decoder.embed.weight.cpu().detach().numpy()
    
    # Get embeddings of chosen words
    y1_emb = np.expand_dims(embedding_matrix[vocab_word2idx[y1]], axis=0)
    y2_emb = np.expand_dims(embedding_matrix[vocab_word2idx[y2]], axis=0)
    x2_emb = np.expand_dims(embedding_matrix[vocab_word2idx[x2]], axis=0)
    
    x1 = y1_emb - y2_emb + x2_emb
    
    # Calculate cosine similarity
    cos_sim = sklearn.metrics.pairwise.cosine_similarity(x1, embedding_matrix)
    
    # Return topk closest words
    closest_k_ind = cos_sim[0].argsort()[-20:][::-1]
    closest_k_words = []
    count = 0    
    for i in closest_k_ind:        
        # Ignore input vectors
        if vocab_idx2word[i] == y1 or vocab_idx2word[i] == y2 or vocab_idx2word[i] == x2:
            continue
        closest_k_words.append([vocab_idx2word[i], cos_sim[0][i]])
        
        # Break if reaches topk
        count += 1
        if count == topk:
            break
        
    return closest_k_words

def cosine_similarity(decoder,vocab,x1,topk=3):
    # Vocab indices
    vocab_word2idx = vocab.word2idx
    vocab_idx2word = vocab.idx2word
    
    # Extract embedding matrix from decoder
    embedding_matrix = decoder.embed.weight.cpu().detach().numpy()
    
    # Get embeddings of chosen words
    x1 = np.expand_dims(embedding_matrix[vocab_word2idx[x1]], axis=0)
    
    # Calculate cosine similarity
    cos_sim = sklearn.metrics.pairwise.cosine_similarity(x1, embedding_matrix)
    
    # Return topk closest words
    closest_k_ind = cos_sim[0].argsort()[-topk:][::-1]
    closest_k_words = []
    for i in closest_k_ind:
        closest_k_words.append(vocab_idx2word[i])
        
    return closest_k_words

def plot_graphs(num_epochs, train_losses, train_acc, bleu2, bleu3, bleu4,val_meteor):
    plt.figure(figsize=(18,6))

    plt.subplot(131)
    plt.title('Acc vs Epoch')
    plt.plot(range(1, num_epochs+1), train_acc, label='train_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()

    plt.subplot(132)
    plt.title('Loss vs Epoch')
    plt.plot(range(1, num_epochs+1), train_losses, label='train_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    
    plt.subplot(133)
    plt.title('Metric Score vs Epoch')
    plt.plot(range(1, num_epochs+1), bleu2, label='Bleu2')
    plt.plot(range(1, num_epochs+1), bleu3, label='Bleu3')
    plt.plot(range(1, num_epochs+1), bleu4, label='Bleu4')
    plt.plot(range(1, num_epochs+1), val_meteor, label='Meteor')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Score')
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    
    fname = 'graph.png'
    
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

def train_epoch(train_loader,device,encoder,decoder,enc_optimizer,dec_optimizer,criterion,vocab,enc_scheduler=None,dec_scheduler=None,view_train_captions=False):
    epoch_loss = 0
    epoch_correct = 0
    epoch_words = 0
    for i, data in enumerate(tqdm(train_loader)):   
        # zero the gradients
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()

        # set decoder and encoder to train mode
        encoder.train()
        decoder.train()

        # Set mini batch dataset
        images, captions, lengths = data
        images, captions = images.to(device), captions.to(device)

        # Forward
        encoded_images = encoder(images)
        predictions, captions, decode_lengths, alphas = decoder(encoded_images, captions, lengths)
        targets = captions[:, 1:]        


        if view_train_captions == True and i%2000 == 0: # View train captions
            top_predictions = predictions.argmax(dim=2) # Select prediction with highest probability
            print("train, epoch={}, i={}".format(epoch,i))
            for sentence in range(len(top_predictions)):
                print("\nPrediction:", end=" ")
                for word in range(len(top_predictions[sentence])):          
                    print(vocab.idx2word[top_predictions[sentence][word].cpu().detach().item()], end=" ")
                print("\nTarget:", end=" ")
                for word in range(len(top_predictions[sentence])):
                    print(vocab.idx2word[targets[sentence][word].cpu().detach().item()], end=" ")
                break # View only 1 train caption
        
        
        # Pack scores and targets as batch 
        batch_scores, _,_,_ = pack_padded_sequence(predictions, decode_lengths, batch_first=True)
        batch_targets, _,_,_ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate accuracy
        top_batch_scores = batch_scores.argmax(dim=1)
        correct = (top_batch_scores == batch_targets).sum().cpu().detach().item()
        words = len(batch_scores)        

        # Calculate loss
        loss = criterion(batch_scores, batch_targets)
        # Backward and optimize
        loss.backward()
#         tqdm.write(_term_move_up()+str(loss.item()))

        dec_optimizer.step()
        #train pretrained stuff only after the first epoch
        enc_optimizer.step()
        #update lr
        dec_scheduler.step()
        enc_scheduler.step()

        epoch_loss += loss.item()
        epoch_correct += correct
        epoch_words += words 

    return epoch_loss,epoch_correct,epoch_words


def val_epoch(val_loader,device,encoder,decoder,vocab,epoch,enc_scheduler=None,dec_scheduler=None,view_val_captions=False):
    """
    some schedulers can be called in validation
    """
    epoch_loss = 0
    epoch_correct = 0
    epoch_words = 0
    bleu2 = 0
    bleu3 = 0
    bleu4 = 0
    meteor = 0
    
    encoder.eval()
    decoder.eval()
    
    for idx,data in enumerate(tqdm(val_loader)):
        images,targ_corpus,meteor_reference = data
        images = images.to(device)
        encoded_images = encoder(images)
        predictions,lengths1,alphas = decoder.inference(encoded_images)
        predicted = predictions.argmax(dim=2)
        pred_corpus = [[] for i in range (len(predicted))]
        meteor_hypothesis = list()
        for i in range(len(predicted)):
            meteor_str = ""
            for j in range(int(lengths1[i].item())):
                term = vocab.idx2word[predicted[i][j].cpu().detach().item()]
                if term == "<end>":
                    break
                pred_corpus[i].append(term)
                meteor_str += term + " "
            meteor_hypothesis.append(meteor_str[:-1])
        if view_val_captions==True and idx%500==0:
            print(pred_corpus[0])
            print(targ_corpus[0])
            

        bleu2 += bleu_score(pred_corpus, targ_corpus,max_n=2,weights=[0.5]*2) #bleu2
        bleu3 += bleu_score(pred_corpus, targ_corpus,max_n=3,weights=[1/3]*3) #bleu3
        bleu4 += bleu_score(pred_corpus, targ_corpus,max_n=4,weights=[0.25]*4) #bleu4 
        temp_meteor = 0
        for i in range(len(pred_corpus)):
            temp_meteor += meteor_score.meteor_score(meteor_reference[i], meteor_hypothesis[i], alpha=0.85, beta=0.2, gamma=0.6) #meteor
        meteor += temp_meteor/len(pred_corpus)       
    
    bleu2 = bleu2/(idx+1)
    bleu3 = bleu3/(idx+1)
    bleu4 = bleu4/(idx+1)
    meteor = meteor/(idx+1)
    
#     print("bleu2",bleu2/(idx+1))
#     print("bleu3",bleu3/(idx+1))
#     print("bleu4",bleu4/(idx+1))


    return bleu2,bleu3,bleu4,meteor

def main(args):    
    # get losses for visualization
    train_losses = list()
    val_losses = list()
    train_acc = list()
    val_acc = list()
    val_bleu2 = list()
    val_bleu3 = list()
    val_bleu4 = list()
    val_meteor = list()

    # data paths
    TRAIN_IMG_PATH = args.train_img_path
    VAL_IMG_PATH = args.val_img_path
    TRAIN_JSON_PATH = args.train_json_path
    VAL_JSON_PATH = args.val_json_path
    VOCAB_PATH = args.vocab_path
    GLOVE_EMBED_PATH = args.glove_embed_path

    print(VOCAB_PATH)

    # Load vocab wrapper
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

    transformation = transforms.Compose([transforms.Resize((224,224)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])

    # Build dataloaders
    batch_size = 32
    TRAIN_LOADER = {'root':TRAIN_IMG_PATH, 'json':TRAIN_JSON_PATH, 'vocab':vocab, 'batch_size':batch_size, 'shuffle':True, 'num_workers':4,'transform':transformation}
    VAL_LOADER_UNIQUE = {'root':VAL_IMG_PATH, 'json':VAL_JSON_PATH, 'batch_size':16, 'shuffle':False,'transform':transformation, 'num_workers':4} #root, json, transform, batch_size, shuffle, num_workers
    train_loader = get_loader(**TRAIN_LOADER)
    val_loader = get_loader_unique(**VAL_LOADER_UNIQUE)
    print(len(train_loader))
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # Build models
    image_dim = 2048
    embed_size = 300
    hidden_size = 512
    vocab_size = len(vocab)

    encoder = CNNfull().to(device)
    decoder = DecoderRNN(image_dim,embed_size,hidden_size,vocab_size).to(device)

    decoder.load_embeddings(torch.load(GLOVE_EMBED_PATH).to(device))
    
    # Loss and optimizer
    #initial lr is chosen based on maxlr/50
    # maxlr is chosen to be maximum lr that gives lowest loss after ~500 batches
    enc_optimizer_lr = 1e-4 
    dec_optimizer_lr = 0.002 #doesnt not matter in onecyclelr

    # weight decay values copied from imagenet and seq2seq models
    #small lr as it is pretrained
    enc_optimizer = optim.SGD(encoder.parameters(),lr=enc_optimizer_lr,momentum=0.9,weight_decay=5e-5)
    #large learning rates for decoder
    dec_optimizer = optim.SGD(decoder.parameters(),lr=dec_optimizer_lr,momentum=0.9,weight_decay=1e-7)
    criterion = nn.CrossEntropyLoss()

    # Train the models
    num_epochs = 2
    view_train_captions = False # View train captions
    view_val_captions = False # View val captions

    dec_scheduler = optim.lr_scheduler.OneCycleLR(dec_optimizer,max_lr=0.1,steps_per_epoch=len(train_loader),epochs=num_epochs)
    enc_scheduler = optim.lr_scheduler.OneCycleLR(enc_optimizer,max_lr=0.005,steps_per_epoch=len(train_loader),epochs=num_epochs)
    
    for epoch in range(num_epochs):
        train_loss = 0    
        val_loss = 0
        train_correct = 0
        train_words = 0
        val_correct = 0
        val_words = 0

        train_loss,train_correct,train_words = train_epoch(train_loader,device,encoder,decoder,enc_optimizer,dec_optimizer,criterion,vocab,enc_scheduler=enc_scheduler,dec_scheduler=dec_scheduler,view_train_captions=view_train_captions)
        
        bleu2,bleu3,bleu4,meteor = val_epoch(val_loader,device,encoder,decoder,vocab,epoch,view_val_captions=view_val_captions)

        average_train_loss = train_loss / (len(train_loader))
        train_losses.append(average_train_loss)

        average_train_acc = train_correct / train_words
        train_acc.append(average_train_acc)
        
        val_bleu2.append(bleu2)
        val_bleu3.append(bleu3)
        val_bleu4.append(bleu4)
        val_meteor.append(meteor)

        if True:
            torch.save(encoder.state_dict(), './weights/g_encoder_weights_epoch{}_bleu{:.5f}.pth'.format(epoch, bleu4))
            torch.save(decoder.state_dict(), './weights/g_decoder_weights_epoch{}_bleu{:.5f}.pth'.format(epoch, bleu4))
            print("Weights saved at epoch {}".format(epoch))

        print("Epoch: {}, Train Loss: {:.5f}, Train Acc: {:.3f}, Val Bleu2: {:.3f}, Val Bleu3: {:.3f}, Val Bleu4: {:.3f}, Val Meteor: {:.3f}".format(epoch, average_train_loss, average_train_acc, bleu2, bleu3, bleu4,meteor))
    
    # Plot graphs of losses and metrics
    plot_graphs(num_epochs, train_losses, train_acc,val_bleu2,val_bleu3,val_bleu4,val_meteor)    
    
    # Print closest k words to analogy
    print(cosine_similarity_analogy(decoder,vocab,y1='man',y2='woman',x2='girl',topk=3)) 
    print(cosine_similarity_analogy(decoder,vocab,y1='beef',y2='cow',x2='pig',topk=3)) 
    print(cosine_similarity_analogy(decoder,vocab,y1='woman',y2='man',x2='businessman',topk=3)) 
    print(cosine_similarity_analogy(decoder,vocab,y1='phone',y2='laptop',x2='macbook',topk=3))
    print(cosine_similarity_analogy(decoder,vocab,y1='man',y2='woman',x2='daughter',topk=3))
    print(cosine_similarity_analogy(decoder,vocab,y1='u.s.',y2='england',x2='london',topk=3))
    
    # Print closest k words to word
    closest_k_words = cosine_similarity(decoder,vocab,x1='holiday',topk=3)  
    print(closest_k_words)


if __name__ == '__main__':
    import os
    import config

    if not os.path.isdir('./weights'):
        os.mkdir('weights')
    
    parser = argparse.ArgumentParser()    
    # Data parameters
    parser.add_argument('--vocab_path', type=str, default=config.VOCAB_PATH, help='path for vocabulary wrapper')
    parser.add_argument('--train_img_path', type=str, default=config.TRAIN_IMG_PATH, help='path for train images')
    parser.add_argument('--val_img_path', type=str, default=config.VAL_IMG_PATH, help='path for val images')
    parser.add_argument('--train_json_path', type=str, default=config.TRAIN_JSON_PATH, help='path for train json')
    parser.add_argument('--val_json_path', type=str, default=config.VAL_JSON_PATH, help='path for val json')    
    parser.add_argument('--glove_embed_path', type=str, default=config.GLOVE_EMBED_PATH, help='path for glove embeddings') 
    args = parser.parse_args(args=[])
    print(args)
    main(args)