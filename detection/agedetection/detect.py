
from PIL import Image
pillow_image = Image.open(r"./image.jpg")
# pillow_image.show()

import cv2
import numpy as np

opencv_image = cv2.cvtColor(np.array(pillow_image), cv2.COLOR_RGB2BGR)
# bw_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Image', bw_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
faceNet=cv2.dnn.readNet(faceModel,faceProto)


def detect_face(image):
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    blob=cv2.dnn.blobFromImage(opencv_image, 1.0, (300, 300), [104, 117, 123], True, False)

    faceNet.setInput(blob)
    detections=faceNet.forward()

    conf_threshold=0.7
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([opencv_image  .shape[1], opencv_image .shape[0], opencv_image  .shape[1], opencv_image .shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            # Draw the bounding box and confidence on the image
            cv2.rectangle(opencv_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            text = f"{confidence * 100:.2f}%"
            cv2.putText(opencv_image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))

pillow_image = Image.open(r"./image.jpg")
output_image = detect_face(pillow_image)
output_image.show()

cv2.imshow("Face Detection", opencv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()