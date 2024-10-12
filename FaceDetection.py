import cv2
#imorted Computer Vision for Python 
#CV-Python is just a wrapper for original Computer Vision 
from tkinter import filedialog 
# tkinter lib is used to allow the user to select an image file 

path=filedialog.askopenfilename(initialdir="/", title="Select Image",filetypes=(("JPEG",".jpg"),("all files","*.*")))
# path var stores the file path selected by user  
Bot=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#Creating a classifier to train on haarcascade frontal face default to detect faces in images 

Test=cv2.imread(path,1)
#

Test_grey=cv2.cvtColor(Test,cv2.COLOR_BGR2GRAY)
Results=Bot.detectMultiScale(Test_grey,scaleFactor=1.2,minNeighbors=5)

for a,b,c,d in Results:
	Test=cv2.rectangle(Test,(a,b),(a+c,b+d),(0,255,0),3)

Resized_Result=cv2.resize(Test,(int(Test.shape[1]/2),int(Test.shape[0]/2)))
cv2.imshow("Final",Resized_Result)
cv2.waitKey(0)
cv2.destroyAllWindows()
