import cv2

trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)
while True:
    successful_frame_read,frame = webcam.read()

# img = cv2.imread('mee.png')

    grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),6)

    cv2.imshow('img', frame)
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break

# print(face_coordinates)
print("complete")
