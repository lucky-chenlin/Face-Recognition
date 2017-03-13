# coding:utf-8
import os as os
import numpy as np
import cv2

cv2.namedWindow("window1")
cap=cv2.VideoCapture(0)
i=1
j=1
move_text={'':'','1':'谌林 ','2':'金鹏','3':'叶杨'}
classfier=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
imageNames = os.listdir("date")
images = []
for name in imageNames:
	images.append(cv2.imread("date" + "/" + name, cv2.IMREAD_GRAYSCALE))
labels = list(range(len(images)))
recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.train(images, np.array(labels))
recognizer.save("recognizer.xml")
while True:
	success,frame=cap.read()
	size=frame.shape[:2]
	image=np.zeros(size,dtype=np.float16)
	image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	cv2.equalizeHist(image,image)
	divisor=8
	h,w=size
	minSize=(w/divisor,h/divisor)
	key=cv2.waitKey(1) & 0xFF
	faceRects=classfier.detectMultiScale(image,1.2,2,cv2.CASCADE_SCALE_IMAGE,minSize)  
	for (x, y, w, h) in faceRects:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (200, 255, 0), 2)
	if len(faceRects) != 0:
		faceRects[:, 2:] += faceRects[:, :2]
	for (x, y, w, h) in faceRects:
		cut = frame[y:h, x:w]
		cut = cv2.resize(cut,(256,256))
		cut=cv2.cvtColor(cut,cv2.COLOR_BGR2GRAY)
		predict_image = np.array(cut,'uint8')
		nbr_predicted = recognizer.predict(predict_image)
		name = imageNames[nbr_predicted]
		name = name[:-7]
		move=move_text[name]		
		print move
		#cv2.putText(frame, move, (30,30), 0, 0.5, (200,255,0),2)
	cv2.imshow("window1", frame)
	if key == ord("q"):
		break
cv2.destroyWindow("window1")

