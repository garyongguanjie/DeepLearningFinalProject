# DeepLearningFinalProject
Coco2014 image captioning with visual attention
# Deploy gui
Ensure [final weights](#final-weights) are downloaded into `./weights` folder
```
pip install -r requirements.txt
python gui.py
```
## Architecture
Almost the same as paper in [show attend and tell](https://arxiv.org/abs/1502.03044).\
Image resize to 224x224 no crop.\
CNN: resnet50 pretrained on imagenet (small finetuning) output feature map is 2048x7x7\
LSTM: Hidden size: 512 \
Word Embeddings size pretrained on Glove(final experiment): 300\
Attention Scoring function: Bahdanau Attention Vt*tanh(W1Q+W2K)\
For faster convergence these tricks are used:\
[Multisample dropout](https://arxiv.org/abs/1905.09788) used in final fc layer as well as on attention layers.\
[OneCyleLr](https://arxiv.org/abs/1708.07120) to train with high learning rates
## How to train
**Ensure that you are in the `scratch` folder.**\
The code below dumps binary vocab dictionary.
```
git clone https://github.com/garyongguanjie/DeepLearningFinalProject.git
cd DeepLearningFinalProject
python build_vocab.py
```
Download Glove embeddings
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/EYRfz8CJmNFEqW9FtA6dTRABzhWQuTNubo6j_mzfKL1tEg?e=Tdm2eb&download=1" -O glove_embeddings.pth
```
To train
```
python train.py --vocab_path VOCAB_PATH --train_img_path TRAIN_IMG_PATH --val_img_path VAL_IMG_PATH --train_json_path TRAIN_JSON_PATH --val_json_path VAL_JSON_PATH --glove_embed_path GLOVE_EMBED_PATH
```
## Final Weights
5 epochs\
Glove embeddings used\
Optimizer:\
Encoder SGD momentum=0.9 weight_decay=5e-5 \
Decoder SGD momentum=0.9,weight_decay=1e-7 \
Scheduler \
Encoder OneCycleLR initial lr 0.0002 maxlr = 0.005 \
Decoder OneCycleLr initial lr 0.004 maxlr = 0.1\
Bleu4 score:24.8\
Download final weights
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/EUpurGS1mXxAg38s8lkAUb8BF80pzSB_Su6TQ6cbCBYXxw?e=SqWtLN&download=1" -O final_weights.zip
```
### First try Weights
3 epochs\
Optimizer:\
Encoder SGD momentum=0.9 weight_decay=5e-4\
Decoder SGD momentum=0.9,weight_decay=1e-6\
Scheduler\
Encoder OneCycleLR initial lr 0.0002 maxlr = 0.005\
Decoder OneCycleLr initial lr 0.004 maxlr = 0.1\
Bleu4 score:24.7\
Download weights from first attempt
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/EUE8VQN6j7dNrRyhPLoCVFkBXYyRoQgcicrRQM_PhxYslg?e=xS0idk&download=1" -O weights.zip
unzip weights.zip
```
