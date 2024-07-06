# Object Detector Web App

This web application utilizes the YOLOv3 model to detect objects within uploaded images. Developed using the Streamlit framework, the app allows users to adjust the confidence threshold for object detection and view the detected objects highlighted in the image.

## Features

- **Upload Images**: Users can upload images in PNG or JPEG format.
- **Adjust Confidence Threshold**: A slider to control the confidence level for object detection.
- **Detect Objects**: The app identifies and classifies objects such as cars, bicycles, and people in the uploaded images.
- **Interactive Interface**: User-friendly interface with buttons to initiate object detection and adjust settings.

## How to Use

1. **Upload an Image**: Click the "Browse files" button or drag and drop an image file into the upload area.
2. **Adjust Confidence Threshold**: Use the slider to set the desired confidence level for detecting objects.
3. **Identify Objects**: Click the "Identify the objects" button to process the image and display the detected objects.
4. **Show image with objects in boxes**: click the "Show image with objects in boxes" to see the annotated image.           
  
## Demo

You can try the web app [here](https://mazenobjectdetection.streamlit.app/).

## Technologies Used

- **Streamlit**: For creating the web interface.
- **OpenCV libaray and YOLOv3 model**: For object detection.
- **Python**: Main programming language used.
