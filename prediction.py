from PIL import Image
from io import BytesIO
import numpy as np
from keras.applications.inception_v3 import preprocess_input
import tensorflow as tf
import pickle
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing import image as img
from keras.applications.inception_v3 import InceptionV3

input_shape = (299, 299)

def load_model():
    model = tf.keras.models.load_model('./model_1_epoch.h5')
    ixtoword = pickle.load(open('./ixtoword.pkl', 'rb'))
    max_length = pickle.load(open('./max_length.pkl', 'rb'))
    wordtoix = pickle.load(open('./wordtoix.pkl', 'rb'))
    model_img = tf.keras.models.load_model('./img_model.h5')
    return model, ixtoword, max_length, wordtoix, model_img

model, ixtoword, max_length, wordtoix, model_img = load_model()

def read_image(image_encoded):
    image = Image.open(BytesIO(image_encoded))
    return image

def preprocess(image: Image.Image):
    image = image.resize(input_shape)
    x = img.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def encode(image):
    image = preprocess(image) 
    fea_vec = model_img.predict(image) 
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec



def predict(image: np.ndarray):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break

    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final
