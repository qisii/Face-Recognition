import cv2

#to capture video
#the  0 means selecting default from your source which is the computer's webcam
cap = cv2.VideoCapture(0)
#using Harr Casscade detection in OpenCV
#using and training a classifier to detect images with faces
#it is called the positive images which contains the images you desire to detect
#positive image is the images with faces that are detected
#while, negative images is the images with no face detected

#when the classifier is trained it can be applied to a region of interest
#the classifier outputs 1 if the region is likely to show the object and 0 otherwise.
#from a repository, I'm using a trained classifier to detect a face

#calling the trained classifier file
cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:

	#reads the captured image or video
	_, frame = cap.read()
	#converting image to grayscale image
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#this declared variable id to detect the faces inside the image
	detections = cascade_classifier.detectMultiScale(gray, 1.2, 5)

	#The rectangle when a face is detected
	if(len(detections) > 0):
		(x,y,w,h) = detections[0]
		frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(34,139,34),2)

	#display the output
	cv2.imshow('frame', frame)
	#this will exit or close the frame when key 'Esc' is pressed
	e = cv2.waitKey(30) & 0xFF 
	if e == 27:
		break

#when everything is done, releases the capture
cap.release()
cv2.destroyAllWindows()