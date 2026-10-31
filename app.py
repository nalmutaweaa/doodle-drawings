import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.title("Doodle Drawing Search")

st.write("Draw something and find top similar doodles!")

# Brush settings
brush_width = st.slider("Brush Size", 1, 30, 8)

# Drawing canvas
canvas = st_canvas(
    stroke_width=brush_width,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    update_streamlit=True,
    key="canvas"
)

# Capture the drawing
if canvas.image_data is not None:
    st.image(canvas.image_data)
