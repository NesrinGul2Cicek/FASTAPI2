from src.utils.utilities import *
import os
import tensorflow as tf
import numpy as np
from PIL import Image

SAVE_LOCATION = os.getcwd() + "/resources/"
IMAGE_SHAPE = (224, 224)

# Globalde bir kez yüklensin:
model = None
labels = None


def load_model():
    global model
    if model is None:
        model_path = "src/pred/models/trafic_signs_model.h5"  # KENDİ MODEL YOLUNU BURAYA YAZ
        model = tf.keras.models.load_model(model_path)
    return model


def load_labels():
    global labels
    if labels is None:
        labels_path = os.path.join(SAVE_LOCATION, 'ImageNetLabels.txt')
        if not os.path.exists(labels_path):
            dir_check(SAVE_LOCATION)  # klasörü yoksa oluştur
            url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
            labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', origin=url)
        with open(labels_path, 'r') as f:
            labels = np.array([line.strip() for line in f.readlines()])
    return labels


def preprocess_img(img: Image.Image):
    img = img.resize(IMAGE_SHAPE)
    img = np.array(img) / 255.0
    return img


def tf_predict(img_original):
    img = preprocess_img(img_original)
    model = load_model()
    result = model.predict(img[np.newaxis, ...])
    predicted_class = tf.math.argmax(result[0], axis=-1)
    scores = tf.nn.softmax(result[0])
    probability = np.max(scores)

    imagenet_labels = load_labels()
    predicted_class_name = imagenet_labels[predicted_class]

    return {
        "predicted_label": predicted_class_name,
        "probability": probability.item()
    }
