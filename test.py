import cv2
import numpy as np
from tensorflow import keras
model=keras.models.load_model("character.h5")

faceCascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

vid = cv2.VideoCapture(0)
while True:
    ret,frame = vid.read()
    cv2.imshow("frame",frame)
    # frame=cv2.imread("test.jpg")
    imagenGrises = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imagenRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    faces = faceCascade.detectMultiScale(
        imagenGrises,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(60, 60)
    )
    #por cada cara detectada pintar un cuadro
    print(len(faces))
    for (x, y, w, h) in faces:
        rostro=cv2.resize(frame[y:y+h,x:x+w],(64,64))
        rostroRGB=cv2.resize(imagenRGB[y:y+h,x:x+w],(64,64))
        rostroRGB=rostroRGB/255.0
        rostro=rostro/255.0
        classes=["savory","unsavory"]
        output=model.predict(np.array([rostroRGB]))

        print(output)
        output=np.argmax(output ,axis=1)
        print(output)
        cv2.putText(rostro,classes[output.squeeze()],(0,60),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1,2)
        cv2.imshow("image",rostro)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

