from flask import Flask, request, jsonify, send_file, render_template
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STITCHED_FOLDER = 'stitched'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STITCHED_FOLDER'] = STITCHED_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(STITCHED_FOLDER):
    os.makedirs(STITCHED_FOLDER)

def stitch_images(images):
    stitcher = cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(images)
    if status == cv2.Stitcher_OK:
        return stitched
    else:
        return None

def remove_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return image[y:y+h, x:x+w]
    else:
        return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    images = []
    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        image = cv2.imread(filepath)
        if image is not None:
            images.append(image)

    if len(images) < 2:
        return jsonify({'error': 'Need at least two images to stitch'}), 400

    stitched = stitch_images(images)
    if stitched is None:
        return jsonify({'error': 'Image stitching failed'}), 500

    cropped = remove_noise(stitched)
    stitched_filename = os.path.join(app.config['STITCHED_FOLDER'], 'stitched.jpg')
    cv2.imwrite(stitched_filename, cropped)

    return send_file(stitched_filename, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
