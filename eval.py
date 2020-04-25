import config
import pickle
from data_loader import get_loader,CocoDataset, get_loader_unique
from build_vocab import Vocabulary
from torchtext.data.metrics import bleu_score
from MyCocoDataset import MyCocoCaptions
from tqdm import tqdm
from nltk.translate import meteor_score
import torch
import torchvision.transforms as transforms
import nltk
from torch import nn,optim
from model import CNNfull,DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
from train import val_epoch
nltk.download('wordnet')

with open(config.VOCAB_PATH, 'rb') as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)

def evaluate(encoder_model_path,decoder_model_path):
    transformation = transforms.Compose([transforms.Resize((224,224)), 
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                        ])

    VAL_LOADER_UNIQUE = {'root':config.VAL_IMG_PATH, 'json':config.VAL_JSON_PATH, 'batch_size':16, 'shuffle':False,'transform':transformation, 'num_workers':4}
    val_loader_unique = get_loader_unique(**VAL_LOADER_UNIQUE)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    encoder = CNNfull(2048)
    encoder.to(device)
    decoder = DecoderRNN(2048,300,512,vocab_size)
    decoder.to(device)
    encoder.load_state_dict(torch.load(encoder_model_path))
    decoder.load_state_dict(torch.load(decoder_model_path))
    encoder.eval()
    decoder.eval()

    bleu2,bleu3,bleu4,meteor = val_epoch(val_loader_unique,device,encoder,decoder,vocab,0,enc_scheduler=None,dec_scheduler=None,view_val_captions=False)
    print(f'Bleu2 score:{bleu2}')
    print(f'Bleu3 score:{bleu3}')
    print(f'Bleu4 score:{bleu4}')
    print(f'Meteor score:{meteor}')

if __name__ == '__main__':
    pass
    # print("Experiment 1")
    # evaluate('./weights/encoder_weights_epoch2_loss6.82144.pth','./weights/decoder_weights_epoch2_loss6.82144.pth')
    # print("Experiment 2")
    # evaluate('./weights/g_encoder_weights_epoch4_bleu0.24715.pth','./weights/g_decoder_weights_epoch4_bleu0.24715.pth')