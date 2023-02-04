
import cv2
#This a simple program is to detect faces in an image------------------------------------
#using Harr Casscade detection in OpenCV
#using and training a classifier to detect images with faces
#it is called the positive images which contains the images you desire to detect
#positive image is the images with faces that are detected
#while, negative images is the images with no face detected

#when the classifier is trained it can be applied to a region of interest
#the classifier outputs 1 if the region is likely to show the object and 0 otherwise.
#from a repository, I'm using a trained classifier to detect a face

#calling the trained classifier file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Read the input image
img = cv2.imread('image1.png')

#converting image to grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#this declared variable id to detect the faces inside the image
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#The rectangle when a face is detected
for (x, y , w ,h) in faces:
	cv2.rectangle(img, (x,y), (x+w, y+h), (34,139,34), 3)

# Display the output
cv2.imshow('img', img)
cv2.waitKey()