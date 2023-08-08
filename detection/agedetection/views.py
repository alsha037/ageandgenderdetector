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

ageProto = "agedetection/age_deploy.prototxt"
ageModel = "agedetection/age_net.caffemodel"
ageNet=cv2.dnn.readNet(ageModel,ageProto)


genderProto = "agedetection/gender_deploy.prototxt"
genderModel = "agedetection/gender_net.caffemodel"
genderNet=cv2.dnn.readNet(genderProto,genderModel)


padding=20
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']






def detect_face(image):
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    blob=cv2.dnn.blobFromImage(opencv_image, 1.0, (300, 300), [104, 117, 123], True, False)

    faceNet.setInput(blob)
    detections=faceNet.forward()

    faceBoxes = []

    conf_threshold=0.7
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([opencv_image.shape[1], opencv_image.shape[0], opencv_image.shape[1], opencv_image.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            # Draw the bounding box and confidence on the image
            faceBoxes.append([startX, startY, endX, endY])
            cv2.rectangle(opencv_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            # text = f"{confidence * 100:.2f}%"
            # cv2.putText(opencv_image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    print(faceBoxes)

    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face=opencv_image[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,opencv_image.shape[0]-1),max(0,faceBox[0]-padding)
                     :min(faceBox[2]+padding, opencv_image.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        cv2.putText(opencv_image, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)


    return Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))