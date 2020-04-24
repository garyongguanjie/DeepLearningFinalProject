from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
import os
from base64 import b64encode
from model_beam_inference import DecoderRNN,CNNfull
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import pickle
import torch
from build_vocab import Vocabulary
import string
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2

def transform_image(image, img_type):
    """
    Summary:
        Transform PIL image to either Tensor or NumPy array
    """
    if img_type == "tensor":
        my_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
        return my_transforms(image).unsqueeze(0)
    elif img_type == "numpy":
        return np.array(image)

def get_attention(image, result, attention_plot):
    attention_list = []
    len_result = len(result)
    for l in range(len_result):
        attention_mask = np.resize(attention_plot[l], (7, 7))
        attention_mask = cv2.resize(attention_mask, (image.shape[1], image.shape[0]))
        attention_mask = ((attention_mask - attention_mask.min()) * (1/(attention_mask.max() - attention_mask.min()) * 255)).astype('uint8')
        attention_img = Image.fromarray(attention_mask)
        attention_img = attention_img.convert('RGB')
        attention_mask = np.array(attention_img)
        combined_img = cv2.addWeighted(image, 0.5, attention_mask, 0.5, 0)
        attention_list.append(Image.fromarray(combined_img))
    return attention_list

# Initialise models
VOCAB_PATH = './data/vocab.pkl'
with open(VOCAB_PATH, 'rb') as f:
    vocab = pickle.load(f)
image_dim = 2048
embed_size = 300
hidden_size = 512
vocab_size = len(vocab)
encoder = CNNfull(pretrained=False)
encoder.load_state_dict(torch.load('./weights/encoder_weights_epoch2_loss6.82144.pth', map_location=torch.device('cpu')))
encoder.eval()
decoder = DecoderRNN(image_dim, embed_size, hidden_size, vocab_size)
decoder.load_state_dict(torch.load('./weights/decoder_weights_epoch2_loss6.82144.pth', map_location=torch.device('cpu')))
decoder.eval()

app = Flask(__name__)
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['webp', 'png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG'])

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # check if the post request has the file part
        # print(request.files['file'].read())
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        f = request.files['file']
        # if user does not select file, browser also submits an empty part without filename
        if f.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if f and allowed_file(f.filename):
            # convert file that to bytes
            img_bytes = f.read()
            image = Image.open(io.BytesIO(img_bytes))
            image = Image.fromarray(np.squeeze(np.array(image)))
            # check image has 1, 3 or 4 channels
            if len(np.array(image).shape) == 2:
                image = image.convert('RGB')
            elif np.array(image).shape[-1] == 3:
                pass
            elif np.array(image).shape[-1] == 4:
                image = image.convert('RGB')
            else:
                flash('Number of image channels not 1, 3 or 4 or image format incorrect')
                return redirect(request.url)
            # do caption prediction
            tensor = transform_image(image, "tensor")
            img_feat = encoder(tensor)
            beam_width = int(request.form.get('list'))
            predictions, lengths, alphas = decoder.inference(img_feat, beam_width=beam_width)
            predicted = predictions.argmax(dim=2)
            caption_list = []
            for i in range(len(predicted)):
                for j in range(int(lengths[i].item())):
                    word = vocab.idx2word[predicted[i][j].item()]
                    if j == 0:
                        caption_list.append(string.capwords(word))
                    elif word != "<end>" and word != ".":
                        caption_list.append(word)
            caption = ""
            for i in range(len(caption_list)):
                if i < len(caption_list) - 1:
                    caption += caption_list[i] + " "
                else:
                    caption += caption_list[i]
            img_numpy = transform_image(image, "numpy")
            # get list of attention  images
            attention_list = get_attention(img_numpy, caption_list, alphas.detach().cpu().squeeze(0).numpy())
            encoded = b64encode(img_bytes).decode('utf-8')
            mime = f.content_type
            uri = "data:%s;base64,%s" % (mime, encoded)
            attention = ""
            for i in range(len(attention_list)):
                byte_io = io.BytesIO()
                attention_list[i].save(byte_io, mime.split('/')[-1])
                byte_io.seek(0)
                data = byte_io.read()
                encoded = b64encode(data).decode('utf-8')
                attention += "data:%s;base64,%s" % (mime, encoded) + " "
            # pass original image, caption, and attention images to be rendered with template
            return render_template('caption.html', uri=uri, caption=caption, attention=attention)
        elif f and not allowed_file(f.filename):
            flash('Invalid file extension')
            return redirect(request.url)
    elif request.method == 'GET':
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)