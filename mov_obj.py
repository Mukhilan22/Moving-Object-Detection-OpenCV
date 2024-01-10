import cv2 #opencv
import time #delay
import imutils #resize

cam = cv2.VideoCapture(0) #cam id
time.sleep(1) #delay

firstFrame = None #to capture first frame (preferably plain)
area = 500

while True:
    _,img = cam.read() #frames from camera
    text = "Normal"
    img = imutils.resize(img, width=500) #resize
    grayI = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Grey IMG
    gausI = cv2.GaussianBlur(grayI,(21,21),0) #Blurred

    if firstFrame is None:   #Capturing First Frame, Enters this if only in the first while loop
        firstFrame = gausI   #first gausI stored as firstFrame
        continue #loop goes back to while

    imgdiff = cv2.absdiff(firstFrame,gausI)  #DIFF in 0-255 value of each pixel
    #imgDiff is the moving object

    threshI = cv2.threshold(imgdiff, 25,255,cv2.THRESH_BINARY)[1]    #thresh img of moving obj
    threshI = cv2.dilate(threshI, None, iterations=2)
    #cv2.dilate is a function in the OpenCV library that makes certain parts of a binary image look bigger or expand them.
    # when none, default kernel 3x3 is used
    #dilated 2 times

    cnts = cv2.findContours(threshI.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts) #border of big objects

    for c in cnts:
        if cv2.contourArea(c) < area: #make full area
            continue
        (x,y,w,h) = cv2.boundingRect(c)
        # cv2.boundingRect(c) is used to find the bounding rectangle (the smallest rectangle) that encloses a set of points or a contour in an image.
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) #Rectangle drawn
        text = "Moving object detected"

    print(text)
    cv2.putText(img,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    cv2.imshow("CAM FEED",img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()









