import cv2
webcam=cv2.VideoCapture(0)
trained_face_data=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
trained_eye_data=cv2.CascadeClassifier("haarcascade_eye.xml")
trained_smile_data=cv2.CascadeClassifier('haarcascade_smile.xml')
while True:
    _,frame =webcam.read()
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face_coordinates= trained_face_data.detectMultiScale(gray_frame)
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(220,200,50),2)
        roi_gray =gray_frame[y:y+h, x:x+w]  
        roi_frame=frame[y:y+h, x:x+w]
        eye_coordinates= trained_eye_data.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eye_coordinates:
            cv2.rectangle(roi_frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        smile_corrdinates = trained_smile_data.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smile_corrdinates:
            cv2.rectangle(roi_frame, (sx, sy), (sx+sw, sy+sh), (0,0,255), 2)
    cv2.imshow("Webcam",frame)
    key=cv2.waitKey(1)
    # 81 and 113 are ASCII Value of q (upper and lower cases respectively)
    if key==81 or key==113:
        break           
webcam.release()  
cv2.destroyAllWindows() 
