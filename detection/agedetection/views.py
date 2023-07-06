from django.shortcuts import render

from django import forms
import os
import io
import base64
import numpy as np
import cv2
 

class ImageUploadForm(forms.Form):
    image = forms.ImageField()


# Create your views here.

from PIL import Image

def index(request):
   if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            # Save the uploaded image temporarily
            image_path = 'temp_image.jpg'
            with open(image_path, 'wb') as f:
                f.write(image.read())
            # Open the image using Pillow
            pil_image = Image.open(image_path)
            output_image = detect_face(pil_image)

            img_bytes = io.BytesIO()
            pil_image.save(img_bytes, format='JPEG')
            output_image.save(img_bytes, format='JPEG')

            img_data = img_bytes.getvalue()
            # Delete the temporary image file
            # os.remove(image_path)
            # Pass the image to the template for display
            return render(request, './agedetection/display_image.html', {'image': base64.b64encode(img_data).decode('utf-8')})
   else:
            form = ImageUploadForm()

   return render(request, './agedetection/index.html',{'form':form})


faceProto="agedetection/opencv_face_detector.pbtxt"
faceModel="agedetection/opencv_face_detector_uint8.pb"
faceNet=cv2.dnn.readNetFromTensorflow(faceModel,faceProto)
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