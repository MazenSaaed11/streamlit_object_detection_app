import cv2
import numpy as np


def load_yolo(weights_path, config_path, names_path):
    net = cv2.dnn.readNet(weights_path, config_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    return net, output_layers, classes


def detect_objects(image, net, output_layers, classes, confidence_threshold, nms_threshold=0.4):
    height, width, channels = image.shape

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence >= confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    detected_objects = []
    for i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        detected_objects.append({"class": label, "confidence": confidence * 100})

    return detected_objects


def annotate_image(image, net, output_layers, classes, confidence_threshold=0.5, nms_threshold=0.4, font_thickness=2):
    detected_objects = detect_objects(image, net, output_layers, classes, confidence_threshold, nms_threshold)
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1
    annotated_image = image.copy()

    for obj in detected_objects:
        label = obj["class"]
        confidence = obj["confidence"]
        x1, y1, x2, y2 = obj["box"]
        color = (0, 0, 255)
        text_color = (0, 255, 0)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_image, f"{label}: {confidence:.2f}%", (x1, y1 - 5), font, font_scale, text_color,
                    font_thickness)

    return annotated_image
