import os
import gc
import io
import time
import base64
import logging

import numpy as np
from PIL import Image

from flask import Flask, request, send_file, jsonify, render_template, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename

import detect

UPLOAD_FOLDER = '/upload'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

run = detect.run()
net = run.load_model(model_name='u2net')

logging.basicConfig(level=logging.INFO)


# @app.route("/", methods=["GET"])
# def hello():
#     return 'Hello guys'


@app.route("/", methods=['GET', 'POST'])
def upload_file():
    return render_template('upload.html')


@app.route("/remove", methods=["GET", "POST"])
def remove():
    if request.method == "POST":
        print('hello')

        start = time.time()
        logging.info(' Removing time!')
        if 'file' not in request.files:
            return jsonify({'error': 'missing file'}), 400

        if request.files['file'].filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
            return jsonify({'error': 'invalid file format'}), 400
        
        data = request.files['file'].read()
        if len(data) == 0:
            return jsonify({'error': 'empty image'}), 400

        # with open('received.jpg', 'wb') as f:     #Save debug locally
        #     f.write(data)
        
        img = Image.open(io.BytesIO(data))
        
        
        # logging.info(' > saving results...')  #Save mask locally
        # with open('mask.png', 'wb') as f:
        #     f.write(mask)
        
        pred_img = run.process(net, img)
        
        # new_img_scaled = new_img.resize((new_img.size[0] * 3, new_img.size[1] * 3))
        # logging.info(' > saving final image...') #Save result locally
        # new_img.save('final_image.png')
        
        predicted_io = io.BytesIO() #Save to buffer
        pred_img.save(predicted_io, format="PNG")
        predicted_io.seek(0)
        predicted = base64.b64encode(predicted_io.getvalue())
        
        logging.info(f" Predicted in {time.time() - start:.2f} sec")
        
        return render_template('predicted.html', predicted=predicted.decode('ascii'))
        # return send_file(predicted_io, mimetype='image/png') #Send file to client

if __name__ == "__main__":
    os.environ['FLASK_ENV'] = 'development'
    port = int(os.environ.get('PORT', 80))
    app.run(debug=True, host='0.0.0.0', port=port)
