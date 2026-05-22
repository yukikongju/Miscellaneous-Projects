import cv2
import supervision as sv
from inference import get_model
from PIL import Image
import requests
from io import BytesIO


# Load a sample image
url = "https://media.roboflow.com/dog.jpeg"
image = Image.open(BytesIO(requests.get(url).content))

# Initialize RF-DETR
model = get_model("rfdetr-base")
predictions = model.infer(image, confidence=0.5)[0]

# Convert predictions to Supervision format
detections = sv.Detections.from_inference(predictions)

# Annotate
annotated_image = sv.BoxAnnotator().annotate(image.copy(), detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections)

cv2.imshow("", annotated_image)
