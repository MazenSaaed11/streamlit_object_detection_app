import streamlit as st
import cv2
import numpy as np
from PIL import Image
import model

# loading yolov3-tiny model
weights_path = "yolov3-tiny.weights"
config_path = "yolov3-tiny.cfg"
names_path = "coco.names"
net, output_layers, classes = model.load_yolo(weights_path, config_path, names_path)

# layout of the web app
icon, title = st.columns([1, 5])

with icon:
    st.image("appIcon.png", width=100)

with title:
    st.title("Object Detector Web App")

st.markdown('---')

user_image, confidence = st.columns([6, 6])
input_image = user_image.file_uploader("Please upload an image to start detecting...", type=["PNG", "JPEG", "JPG"])
threshold = confidence.slider(label='Confidence Threshold(%):', min_value=0, max_value=100, step=1, value=80)
confidence.caption('Move the slider to choose how certain you want the app to be when detecting objects.')

image_placeholder = st.empty()
if input_image is not None:
    image_placeholder.image(input_image)

show_button = input_image is not None

identify_btn = st.empty()
result = st.empty()
show_boxed_image = st.empty()
boxed_image_placeholder = st.empty()

if 'previous_image' not in st.session_state:
    st.session_state.previous_image = None

if 'show_boxed_image' not in st.session_state:
    st.session_state.show_boxed_image = False

if 'show_results' not in st.session_state:
    st.session_state.show_results = False

def identify_btn_clicked():
    st.session_state.show_results = True


def show_boxed_image_clicked():
    st.session_state.show_boxed_image = True



if input_image is None or input_image != st.session_state.previous_image:
    st.session_state.previous_image = input_image
    st.session_state.show_boxed_image = False
    st.session_state.show_results = False

if show_button:
    identify_btn.button('Identify the objects', on_click=identify_btn_clicked)
    show_boxed_image.button('Show image with objects in boxes', on_click=show_boxed_image_clicked)

if st.session_state.show_results:
    result.image("waiting.gif")
    image = Image.open(input_image)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    detected_objects = model.detect_objects(img_cv, net, output_layers, classes, confidence_threshold=threshold / 100.0)

    if detected_objects:
        detected_info = "### Objects detected:\n"
        for obj in detected_objects:
            detected_info += f"- **Class:** {obj['class']}, **Confidence:** {obj['confidence']:.2f}%\n"
        result.markdown(detected_info)

if st.session_state.show_boxed_image:
    boxed_image_placeholder.image("waiting.gif")
    image = Image.open(input_image)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    annotated_image = model.annotate_image(img_cv, net, output_layers, classes, confidence_threshold=threshold / 100.0)
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    annotated_image_pil = Image.fromarray(annotated_image_rgb)
    boxed_image_placeholder.image(annotated_image_pil)

st.markdown('---')

github_icon, github_link, linkedin_icon, linkedin_link = st.columns([1, 5, 1, 5])

with github_icon:
    st.image("github.png", width=30)

with github_link:
    st.markdown("[My GitHub](https://github.com/MazenSaaed11)")

with linkedin_icon:
    st.image("linkedin.png", width=30)

with linkedin_link:
    st.markdown("[My LinkedIn](https://www.linkedin.com/in/mazen-saaed-383247235/)")
