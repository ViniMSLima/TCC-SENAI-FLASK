from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("flaskr/model5.keras")

bp = Blueprint('json', __name__, url_prefix='/json')

maybeResults = ["bad_blue", "bad_red", "good_blue", "good_red"]

@bp.route('/', methods=['POST'])
def process_images():
    data = request.get_json()
    nImages = len(data['images'])
    wordResults = []

    for img_path in data['images']:
        img_data = np.array([tf.keras.utils.load_img(img_path)])
        wordResults.append(model.predict(img_data))

    results = ""
    for i in range(nImages):
        results += maybeResults[np.argmax(wordResults[i][0])]

    print(results)
    
    return jsonify(results)
