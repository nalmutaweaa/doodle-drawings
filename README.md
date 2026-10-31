# 🎨 Doodle Drawing Search

## Overview
Doodle Drawing Search is a simple image similarity search application built with Streamlit. Users can draw or upload an image, and the app displays the 25 most visually similar doodles from the dataset.

## Dataset
The project uses Google's **Quick, Draw! Dataset**, which contains 28×28 grayscale drawings across different categories.

## How It Works
1. **Preprocessing:** Images are resized, converted to RGB, and prepared for the model.
2. **Feature Extraction:** A pretrained MobileNetV2 model extracts image features as numerical embeddings.
3. **Similarity Search:** NearestNeighbors uses cosine distance to find the 25 most similar drawings.
4. **Results:** The app displays the top 25 matching doodles with their similarity scores.

## Tools & Libraries
- Python
- TensorFlow / Keras (MobileNetV2)
- Scikit-learn (NearestNeighbors)
- NumPy
- Streamlit

## Features
- Draw directly on a canvas.
- Upload an image (PNG or JPG).
- Find the top 25 similar drawings.
- View cosine similarity scores.

## Limitations
The application always returns the closest available drawings, even if the input does not belong to the dataset's categories.

## Dataset Source
[Google Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)
