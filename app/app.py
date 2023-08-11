from flask import Flask, request, render_template, jsonify,Response,url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import pathlib



""" tf.config.experimental.set_config({
    "device_count": {"CPU": 0},
})
 """
app = Flask(__name__)
vgg_model = VGG16()
# restructure the model
vgg_model = Model(inputs=vgg_model.inputs,
                  outputs=vgg_model.layers[-2].output)

# Load your image captioning model
model = load_model('./best_model.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_length = 28
# generate caption for an image


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        print(in_text)
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
    return in_text


@app.route('/',methods =['GET','POST'])
def index():
    # You can create an HTML template for the front-end if needed
    return render_template('index.html', caption="", image_url="")


@app.route('/get_caption', methods=['POST'])
def get_caption():
    # Assuming you are receiving an image file through a form
    image_file = request.files['image']
    # Convert the image file to a path-like object
    path = pathlib.Path(image_file.filename)

    # Preprocess the image
    img = load_img(path, target_size=(224, 224))
    img = img_to_array(img)
    # reshape data for model
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    # preprocess image from vgg
    img = preprocess_input(img)
    # extract features
    feature = vgg_model.predict(img, verbose=0)
    # predict from the trained model

    # Use your model to generate a caption
    # Replace this with actual caption generation code
    caption = predict_caption(model, feature, tokenizer, max_length)
    caption = caption.replace('startseq', '').replace('endseq', '').strip()
    image_url = url_for('static', filename=image_file.filename)
    return render_template('index.html', caption=caption, image_url=image_url)
    
    


if __name__ == '__main__':
    app.run(debug=True)