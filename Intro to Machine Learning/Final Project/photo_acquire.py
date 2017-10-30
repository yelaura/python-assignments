import cv2
import numpy as np
import os
import re
from datetime import datetime, timedelta

image_path = 'photo'
n = 100
start_id = 0

# take a series of images within video
def image_capture(image_path, n, start_id):
    """
    Take a series of images with face from vedio
    
    Input:
    -image_path: path of image folder to store captured images.
    -n: number of images to capture. 
    -start_id: the start number of image used in filename. Default start from 0.
    
    Output:
    -n images with faces stored in designiated image folder.
    """
    # senitize path and create folder if not exists
    image_path = image_path
    if not image_path.endswith('\\'):
        image_path +='\\'
    if not os.path.exists(image_path):
        os.mkdir(image_path)
        
    img_id = start_id
    end_id = img_id+n

    start = False
    oldtime = datetime.now()
    
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    while(img_id<end_id):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # face tracking on gray image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Display the resulting frame  
        for (x,y,w,h) in faces:
            cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('s'):
            start = True
            oldtime = datetime.now()

        # take next image after 0.5 second if face is detected.
        newtime = datetime.now()
        diff = (newtime-oldtime).seconds

        if len(faces) and start and diff>0.5:               
            filename = 'img_%i.jpg'%img_id
            cv2.imwrite(str(image_path)+filename,frame)
            img_id+=1
            oldtime = datetime.now()
        
        if start:
            cv2.putText(gray, "Capture image {0}; last image is {1}".format(img_id, end_id-1), (40, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)
        else:
            cv2.putText(gray, "Press s to start, press q to quit!", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)
        cv2.imshow('frame',gray)
        faces = None

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# capture 100 images with face and store file in folder named 'weiya'
image_capture(image_path, n, start_id)