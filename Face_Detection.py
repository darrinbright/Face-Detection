import cv2
trained_face_data=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img = cv2.imread("img.jpg")


gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
face_coordinates=trained_face_data.detectMultiScale(gray_img)
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                                       #color #thickness(rec) 
 #print(face_coordinates)

cv2.imshow('Title',img)
cv2.waitKey()


