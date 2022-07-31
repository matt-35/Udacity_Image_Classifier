# %%
import os 
import sys 
import json 
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf 
import tensorflow_hub as hub
from argparse import ArgumentParser

def load_model(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    return model

def load_classes(path):

    with open(path, 'r') as f:
        class_names = json.load(f) # Returns Dict

    return class_names

def process_image(image, image_size=224):
    # image is in the form of NumPy array.
    image = tf.image.resize(image, (image_size, image_size)).numpy()
    image /= 255
    return image

def predict(image_path, model, top_k):

    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)

    predictions = model.predict(np.expand_dims(image, axis=0))

    top_k = np.argsort(predictions[0])[-top_k:]
    top_k_predictions = [predictions[0][i] for i in top_k]
    top_k_labels = [str(i + 1) for i in top_k]

    return top_k_predictions, top_k_labels

def plot_predict(image_path, model, k_predictions, class_names):

    image = np.asarray(Image.open(image_path))
    image = process_image(image)


    top_k_predictions, top_k_labels = predict(image_path, model, k_predictions)
    class_labels = [class_names[i] for i in top_k_labels]

    fig, (ax1, ax2) = plt.subplots(figsize = (9,9), ncols=2, constrained_layout=True)
    ax1.imshow(image)
    ax2.barh(np.arange(5), top_k_predictions)
    ax2.set_yticks(np.arange(5))
    ax2.set_yticklabels(class_labels)
    ax2.set_xlim(0,1)
    ax2.set_aspect(0.2)
    ax2.set_title('Class Probability')
    plt.show()

    return 

def parse_args():

    my_parser = ArgumentParser()

    my_parser.add_argument('image')
    my_parser.add_argument('model')
    my_parser.add_argument('--top_k', action='store', type=int, required=False)
    my_parser.add_argument('--category_names', action='store', type=str, required=False)

    args = my_parser.parse_args()

    image = args.image
    model = args.model 
    top_k = args.top_k 
    class_path = args.category_names

    if top_k == None:
        top_k = 5 
    
    if class_path == None:
        class_names = load_classes('label_map.json')
    else:
        class_names = load_classes(class_path)

    return image, model, top_k, class_names

# %%
if __name__ == '__main__':
    
    image_path, model_path, top_k, class_names = parse_args()
    
    model = load_model(model_path)

    plot_predict(image_path, model, top_k, class_names)

# %%
## Example commands:

# python predict.py test_images/wild_pansy.jpg models/my_model.h5
# python predict.py test_images/cautleya_spicata.jpg models/my_model.h5 --top_k 5 --category_names label_map.json