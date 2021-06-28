import cv2
import numpy as np
import os

os.chdir("/home/pi/opencv-3.4.1/data/haarcascades")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/home/pi/FaceRecognition/trainer/trainer.yml')
cascadePath = "/home/pi/opencv-3.4.1/data/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

names = ['Fırat', 'Ayşe', 'Mehmet', 'Arzu', 'Kemal', 'Nazlı']

# görüntüyü başlat 
cam = cv2.VideoCapture(0)
cam.set(3, 640) # görüntü en uzunluğu
cam.set(4, 480) # görüntü boy uzunluğu

# açılır pencerenin minimum en boy oranı yüzü tanımlamak için
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img =cam.read()
    #img = cv2.flip(img, -1) 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

    cv2.imshow('camera',img)

    k = cv2.waitKey(10) & 0xff # esc tuşuna basılır videodan çıkmak için (10.klavye tuşu escdir . )
    if k == 27:
        break
