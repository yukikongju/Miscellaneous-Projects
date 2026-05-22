#  https://supervision.roboflow.com/annotators/


import cv2
from rfdetr import RFDETRBase
from PIL import Image
import supervision as sv

model = RFDETRBase()
model.optimize_for_inference()  # The secret sauce for speed

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # RF-DETR expects RGB (PIL), but OpenCV uses BGR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Predict
    detections = model.predict(pil_image, threshold=0.5)

    # Annotate the BGR frame directly
    annotated_frame = sv.BoxAnnotator().annotate(frame, detections)
    #  annotated_frame = sv.LabelAnnotator().annotate(frame, detections)

    cv2.imshow("RF-DETR Real-Time", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
