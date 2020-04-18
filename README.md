# DeepLearningFinalProject
Coco2014 image captioning with visual attention

## Dump feature maps into files
This is done without finetuning. As 1 epoch on the forward pass takes about ~10 minutes.\
**Ensure that you are in the `scratch` folder.**\
The code below dumps torch binary features and binary vocab dictionary.
```
git clone https://github.com/garyongguanjie/DeepLearningFinalProject.git
cd DeepLearningFinalProject
python makeFeatureDump.py
python build_vocab.py
```
See untitled.ipynb
* Current architecture uses
Image -> resize to 224x224 -> resnet34 ->bs,512,7,7-> bs,512,49 -> (trainable) encoder -> bs,49,512 to be passed into decoder with attention
* Decoder is the regular image captioning with attention deocder (see `lec22_attentionmodels.pdf`).
* Key is hidden state. Query is image vectors. Value here is also image vectors. Image vectors are of shape 49,512. Analogy to seq2seq model is that 1 word = 1 channel of the image where 1 channel is of shape 49,1. Attention here is to create a weight for each of the 512 channels and take a weighted mean to output final shape 49,1. This is then concatenated with word embeddings. To be passed as input to LSTM.
* One difference here is that word embeddings are learnt instead of one hot encoded. This is done using `nn.Embedding`. Output is still a softmax function of all possible words.  
* Why `LSTMcell` instead of `LSTM` in decoder?\
Hidden state of each cell is used as key for attention. Cannot do that in regular LSTM when for loop is done for you.\
Hence need to write for loop your self and pass in batch of **a single** word by word and manually pass in hidden states into attention module. 

## TODOS
* Training loop
* Check correctness. Not 100% sure if got mistakes. There might be some misalignment between inputs and outputs?
* Evaluation loop
* Output words
* Bleu Score
* Frontend
### Extras if got time
* Beam search
* Plot attention and show on frontend (definitely many points for this)
