from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
import os
from base64 import b64encode

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

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
            # img_bytes = f.read()
            # caption, attention_plot = get_prediction(image_bytes=img_bytes)
            # json = jsonify({'json': caption, 'attention_plot': attention_plot})
            encoded = b64encode(f.read()).decode('utf-8')
            mime = f.content_type
            uri = "data:%s;base64,%s" % (mime, encoded)
            return render_template('caption.html', uri=uri)
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