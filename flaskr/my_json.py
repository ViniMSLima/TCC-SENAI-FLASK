from io import BytesIO
from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("flaskr/model5.keras")

bp = Blueprint('json', __name__, url_prefix='/json')

maybeResults = ["bad_blue", "bad_red", "good_blue", "good_red"]

@bp.route('/', methods=['POST'])
def process_images():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image_file = request.files['image']

    # Converter o arquivo para um objeto BytesIO
    image_stream = BytesIO(image_file.read())
    
    # Convertendo a imagem para o formato necessário para a previsão
    img = tf.keras.utils.img_to_array(tf.keras.utils.load_img(image_stream, target_size=(224, 224)))  # Ajuste o target_size conforme necessário
    img = np.expand_dims(img, axis=0)
    
    # Realizando a predição com o modelo
    predictions = model.predict(img)
    
    # Convertendo predições para uma resposta
    result = maybeResults[np.argmax(predictions[0])]
    
    return jsonify(result)
