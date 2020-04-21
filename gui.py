from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
import os
from base64 import b64encode
from model import EncoderCNN, DecoderRNN
import torchvision.transforms as transforms
from PIL import Image
import io

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# encoder = EncoderCNN(512, image_dim)
# decoder = DecoderRNN(image_dim, embed_size, hidden_size, vocab_size)

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
            tensor = transform_image(image_bytes=img_bytes)
            # encoded_images = encoder(tensor)
            # preds, captions, decode_lengths, alphas = decoder(encoded_images, captions, lengths)
            # top_predictions = preds.argmax(dim=2)
            # json = jsonify({'json': caption, 'attention_plot': attention_plot})
            encoded = b64encode(img_bytes).decode('utf-8')
            mime = f.content_type
            uri = "data:%s;base64,%s" % (mime, encoded)
            caption = "GG BRO"
            return render_template('caption.html', uri=uri, caption=caption)
        elif f and not allowed_file(f.filename):
            flash('Invalid file extension')
            return redirect(request.url)
    elif request.method == 'GET':
        return render_template('home.html')

# @app.route('/caption', methods=['GET'])
# def show():
#     # encoded = b64encode(request.args.get('image').read())
#     encoded = request.args.get('encoded')
#     mime = request.args.get('mime')
#     uri = "data:%s;base64,%s" % (mime, encoded)

#     return render_template('caption.html', uri=uri)

if __name__ == '__main__':
    app.run(debug=True)