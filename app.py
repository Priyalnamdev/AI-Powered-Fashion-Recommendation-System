import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import os
import streamlit as st

st.header('Fashion Recommendation System')

# Load precomputed features and filenames
Image_features = pkl.load(open(r"C:\Users\HP\Downloads\embeddings.pkl", 'rb'))
filenames = pkl.load(open(r"C:\Users\HP\Downloads\filenames.pkl", 'rb'))

# Feature extraction function
def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)
    return norm_result

# Load ResNet50 model for feature extraction
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Fit NearestNeighbors model
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

# Streamlit file uploader
upload_file = st.file_uploader("Upload Image")
if upload_file is not None:
    upload_dir = r"C:\Users\HP\OneDrive\Desktop\upload"
    os.makedirs(upload_dir, exist_ok=True)  # Ensure the upload directory exists
    file_path = os.path.join(upload_dir, upload_file.name)
    with open(file_path, 'wb') as f:
        f.write(upload_file.getbuffer())
    
    st.subheader('Uploaded Image')
    st.image(file_path)

    # Extract features of uploaded image
    input_img_features = extract_features_from_images(file_path, model)

    # Find nearest neighbors
    distances, indices = neighbors.kneighbors([input_img_features])

    # Display recommended images
    st.subheader('Recommended Images')
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(filenames[indices[0][1]])
    with col2:
        st.image(filenames[indices[0][2]])
    with col3:
        st.image(filenames[indices[0][3]])
    with col4:
        st.image(filenames[indices[0][4]])
    with col5:
        st.image(filenames[indices[0][5]])
