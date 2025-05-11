import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load models
plant_disease_model = tf.keras.models.load_model('backend/models/plant_disease_model.h5')
tree_species_model = tf.keras.models.load_model('backend/models/tree_species_model.h5')

# Load plant disease labels dynamically from file
def load_labels(filepath):
    with open(filepath, "r") as f:
        return [line.strip() for line in f.readlines()]

# Update label mappings
plant_disease_labels = load_labels("backend/plant_disease_labels.txt")  # New file for plant disease labels
tree_species_labels = load_labels("backend/tree_species_labels.txt")

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_plant_disease(image_path):
    img_array = preprocess_image(image_path)
    prediction = plant_disease_model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=-1)[0]

    if predicted_class < len(plant_disease_labels):
        return plant_disease_labels[predicted_class]
    return "Unknown"

def predict_tree_species(image_path):
    img_array = preprocess_image(image_path)
    prediction = tree_species_model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=-1)[0]

    if predicted_class < len(tree_species_labels):
        return tree_species_labels[predicted_class]
    return "Unknown"
