import streamlit as st
import numpy as np

from PIL import Image
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.neighbors import NearestNeighbors


# Load model
@st.cache_resource
def load_model():
    return MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg"
    )


# Load saved dataset
@st.cache_data
def load_data():
    data = np.load("doodle_gallery.npz")

    return data["images"], data["embeddings"]


# Preprocess drawing
def preprocess_drawing(image):

    image = image.convert("RGBA")

    # Handle transparent background
    background = Image.new("RGBA", image.size, "black")
    image = Image.alpha_composite(background, image)

    image = image.convert("L")
    image = image.resize((28, 28))

    return np.array(image)


# Extract embedding
def extract_embedding(doodle, model):

    image = Image.fromarray(doodle.astype("uint8"))
    image = image.resize((160, 160))
    image = image.convert("RGB")

    image = np.array(image, dtype="float32")
    image = preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    return model.predict(image, verbose=0)


# Load resources
model = load_model()
gallery_x, feature_list = load_data()


# Search engine
@st.cache_resource
def load_neighbors(_features):

    neighbors = NearestNeighbors(
        n_neighbors=25,
        metric="cosine"
    )

    neighbors.fit(_features)

    return neighbors


neighbors = load_neighbors(feature_list)


# App
st.title("Doodle Search")

st.write("Draw or upload an image to find similar doodles.")


# Drawing canvas
canvas = st_canvas(
    stroke_color="#FFFFFF",
    background_color="#000000",
    stroke_width=8,
    height=280,
    width=280,
    drawing_mode="freedraw",
    return_image_data=True,
    key="canvas"
)

# Upload image
uploaded_file = st.file_uploader(
    "Or upload an image",
    type=["png", "jpg", "jpeg"]
)


# Search button
if st.button("Search"):

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

    elif canvas.image_data is not None:
        image = Image.fromarray(
            canvas.image_data.astype("uint8")
        )

    else:
        image = None

    if image is None:
        st.warning("Please draw or upload an image.")

    else:
        drawing = preprocess_drawing(image)

        if drawing.max() <= 10:
            st.warning("Please draw something first.")

        else:
            # Show processed image
            st.write("Your Drawing")
            st.image(drawing, width=150, clamp=True)

            # Extract features
            embedding = extract_embedding(drawing, model)

            # Search
            distances, indices = neighbors.kneighbors(embedding)

            similarities = 1 - distances[0]

            # Display results
            st.subheader("Top 25 Similar Drawings")

            for row in range(5):

                cols = st.columns(5)

                for col in range(5):

                    i = row * 5 + col
                    idx = indices[0][i]

                    with cols[col]:
                        img = gallery_x[idx].reshape(28, 28)

                        st.image(
                            img,
                            caption=f"{similarities[i]:.2f}",
                            width=100,
                            clamp=True
                        )
