import cv2
from random import randrange
webcam=cv2.VideoCapture(0)
trained_face_data=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    _,frame =webcam.read()
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face_coordinates= trained_face_data.detectMultiScale(gray_frame)

    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(220,200,50),3)
    cv2.imshow("Webcam",frame)
    key=cv2.waitKey(1)
    if key==81 or key==113:
        break           
webcam.release()    