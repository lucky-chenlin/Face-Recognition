import cv2
import numpy as np
cv2.namedWindow("window1")
cap=cv2.VideoCapture(0)
i=1
j=1
classfier=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
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
		if key== ord('s'):
			cut = frame[y:h, x:w]
			cut = cv2.resize(cut,(256,256))
			name=str(i)+"_"+str(j)+".jpg"
			cv2.imwrite("Faces"+'/'+name,cut)
			if(j<20):
				j+=1
			else:
				while(0xFF & cv2.waitKey(0)!=ord('n')):
					j=1
				j=1
				i+=1	
	cv2.imshow("window1", frame)
	if key == ord("q"):
		break
cv2.destroyWindow("window1")