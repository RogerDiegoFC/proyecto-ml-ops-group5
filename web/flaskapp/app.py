from flask import Flask, request, render_template
import pandas as pd
import os
import cv2
import numpy as np
from tensorflow import keras
import pickle

app = Flask(__name__)

# Cargar el modelo Keras una sola vez (más eficiente que cargarlo en cada request)
model = keras.models.load_model("ml/my_model.keras")

@app.route('/', methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route('/predict', methods=["GET", "POST"])
def predict():
    new_file = request.files['file']
    target_path = os.path.join("upload", new_file.filename)
    new_file.save(target_path)

    image = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    image = data_validation(image)

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]  # clase más probable

    return f"Es un {predicted_class} ;) !!"

def data_validation(image):
    # Redimensionar a 28x28
    image = cv2.resize(image, (28, 28))

    # Normalizar a rango [0,1]
    image = image.astype("float32") / 255.0

    # Añadir dimensión del canal (grises → 1 canal)
    image = np.expand_dims(image, axis=-1)

    # Añadir dimensión del batch
    image = np.expand_dims(image, axis=0)

    return image

if __name__ == '__main__':
    app.run(debug=True, port=5002)
