import pickle
from data_loader import get_loader,CocoDataset
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

def plot_graphs(num_epochs, train_losses, val_losses, train_acc, val_acc):
    plt.figure(figsize=(12,6))

    plt.subplot(121)
    plt.title('Acc vs Epoch')
    plt.plot(range(1, num_epochs+1), train_acc, label='train_acc')
    plt.plot(range(1, num_epochs+1), val_acc, label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()

    plt.subplot(122)
    plt.title('Loss vs Epoch')
    plt.plot(range(1, num_epochs+1), train_losses, label='train_loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    
    fname = 'graph.png'
    
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

def train_epoch(train_loader,device,encoder,decoder,enc_optimizer,dec_optimizer,criterion,vocab,epoch,embed_optimizer=None,enc_scheduler=None,dec_scheduler=None,view_train_captions=False):
    epoch_loss = 0
    epoch_correct = 0
    epoch_words = 0
    for i, data in enumerate(tqdm(train_loader)):        
        # zero the gradients
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        embed_optimizer.zero_grad()

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
        tqdm.write(_term_move_up()+str(loss.item()))
        dec_optimizer.step()

        #train pretrained stuff only after the first epoch
        if epoch != 0:
            enc_optimizer.step()
            embed_optimizer.step()
        #update lr
        dec_scheduler.step()

        epoch_loss += loss.item()
        epoch_correct += correct
        epoch_words += words 

    return epoch_loss,epoch_correct,epoch_words


def val_epoch(val_loader,device,encoder,decoder,criterion,vocab,epoch,enc_scheduler=None,dec_scheduler=None,view_val_captions=False):
    """
    some schedulers can be called in validation
    """
    epoch_loss = 0
    epoch_correct = 0
    epoch_words = 0

    for i, data in enumerate(tqdm(val_loader)):
        # deactivate autograd engine
        with torch.no_grad():
            # set decoder and encoder to eval mode
            encoder.eval()
            decoder.eval()

            # Set mini batch dataset
            images, captions, lengths = data
            images, captions = images.to(device), captions.to(device)

            # Forward
            encoded_images = encoder(images)
            predictions, captions, decode_lengths, alphas = decoder(encoded_images, captions, lengths)
            targets = captions[:, 1:]

            if view_val_captions == True and i%2000 == 0: # View val captions
                top_predictions = predictions.argmax(dim=2) # Select prediction with highest probability
                print("eval, epoch={}, i={}".format(epoch,i))
                for sentence in range(len(top_predictions)):
                    print("\nPrediction:", end=" ")
                    for word in range(len(top_predictions[sentence])):          
                        print(vocab.idx2word[top_predictions[sentence][word].cpu().detach().item()], end=" ")
                    print("\nTarget:", end=" ")
                    for word in range(len(top_predictions[sentence])):
                        print(vocab.idx2word[targets[sentence][word].cpu().detach().item()], end=" ")

            # Pack scores and targets as batch 
            batch_scores, _,_,_ = pack_padded_sequence(predictions, decode_lengths, batch_first=True)
            batch_targets, _,_,_ = pack_padded_sequence(targets, decode_lengths, batch_first=True)       

            # Calculate accuracy
            top_batch_scores = batch_scores.argmax(dim=1)
            correct = (top_batch_scores == batch_targets).sum().cpu().detach().item()
            words = len(batch_scores)
            epoch_correct += correct
            epoch_words += words            

            # Calculate loss
            loss = criterion(batch_scores, batch_targets)
            epoch_loss += loss.item()   

    return epoch_loss,epoch_correct,epoch_words

def main(args):    
    # get losses for visualization
    train_losses = list()
    val_losses = list()
    train_acc = list()
    val_acc = list()

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
    VAL_LOADER = {'root':VAL_IMG_PATH, 'json':VAL_JSON_PATH, 'vocab':vocab, 'batch_size':batch_size, 'shuffle':False, 'num_workers':4,'transform':transformation}
    train_loader = get_loader(**TRAIN_LOADER)
    val_loader = get_loader(**VAL_LOADER)
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
    dec_optimizer_lr = 0.004 #doesnt not matter in onecyclelr
    embed_optimizer_lr = 1e-4

    # weight decay values copied from imagenet and seq2seq models
    #small lr as it is pretrained
    enc_optimizer = optim.SGD(encoder.parameters(),lr=enc_optimizer_lr,momentum=0.9,weight_decay=1e-4)
    #small lr for embed layer as it is pretrained except for a few words
    embed_optimizer = optim.SGD(list(decoder.parameters())[:1],lr=embed_optimizer_lr,momentum=0.9,weight_decay=1e-7)
    #large learning rates for decoder
    dec_optimizer = optim.SGD(list(decoder.parameters())[1:],lr=dec_optimizer_lr,momentum=0.9,weight_decay=1e-7)
    criterion = nn.CrossEntropyLoss()

    # Train the models
    num_epochs = 4
    view_train_captions = False # View train captions
    view_val_captions = True # View val captions

    dec_scheduler = optim.lr_scheduler.OneCycleLR(dec_optimizer,max_lr=0.2,steps_per_epoch=len(train_loader),epochs=num_epochs)
    
    for epoch in range(num_epochs):
        train_loss = 0    
        val_loss = 0
        train_correct = 0
        train_words = 0
        val_correct = 0
        val_words = 0

        train_loss,train_correct,train_words = train_epoch(train_loader,device,encoder,decoder,enc_optimizer,dec_optimizer,criterion,vocab,epoch=epoch,embed_optimizer=embed_optimizer,enc_scheduler=None,dec_scheduler=dec_scheduler,view_train_captions=view_train_captions)
        
        val_loss,val_correct,val_words = val_epoch(val_loader,device,encoder,decoder,criterion,vocab,epoch,view_val_captions=view_val_captions)

        average_train_loss = train_loss / (len(train_loader))
        train_losses.append(average_train_loss)

        average_val_loss = val_loss / (len(val_loader))
        val_losses.append(average_val_loss) 

        average_train_acc = train_correct / train_words
        train_acc.append(average_train_acc)

        average_val_acc = val_correct / val_words
        val_acc.append(average_val_acc)  

        if True:
            torch.save(encoder.state_dict(), './weights/g_encoder_weights_epoch{}_loss{:.5f}.pth'.format(epoch, average_val_loss))
            torch.save(decoder.state_dict(), './weights/g_decoder_weights_epoch{}_loss{:.5f}.pth'.format(epoch, average_val_loss))
            print("Weights saved at epoch {}".format(epoch))

        print("Epoch: {}, Train Loss: {:.5f}, Val Loss: {:.5f}, Train Acc: {:.3f}, Val Acc: {:.3f}".format(epoch, average_train_loss, average_val_loss, average_train_acc, average_val_acc))

    plot_graphs(num_epochs, train_losses, val_losses, train_acc, val_acc)


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

